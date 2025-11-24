import re
import math
import numpy as np
import pandas as pd

from dash import Dash, dcc, html, Input, Output, State
from dash import dash_table
import plotly.graph_objects as go

from chem_sim.simulator import Simulator
from chem_base.reaction import TEReaction


DEFAULT_MECHANISM = """\
A + cat -> cat1
cat1 -> cat + P
cat -> catI
""".strip()

# Defaults for rate constants edited in the table
K_FWD_DEFAULT = 30
K_REV_DEFAULT = 0.0


def parse_mechanism_text(mech_text):
    """
    Returns (reactions, mechanism_lines) where:
      - reactions: list[TEReaction] with id_ = 1..N
      - mechanism_lines: the cleaned reaction strings used for TEReaction
    Skips blank lines and '#' comments.
    """
    lines = []
    for raw in (mech_text or "").splitlines():
        ln = raw.split("#", 1)[0].strip()
        if not ln:
            continue
        # Normalize arrow spacing a bit
        ln = re.sub(r"\s*->\s*", " -> ", ln)
        # Only support irreversible arrow; reversibility is via kN sliders (k_reverse column)
        if "->" not in ln:
            raise ValueError(f"Mechanism line missing '->': '{ln}'")
        lines.append(ln)

    if not lines:
        raise ValueError("No valid mechanism steps found. Please enter at least one line with '->'.")

    reactions = [TEReaction(reaction_string=ln, id_=i + 1) for i, ln in enumerate(lines)]
    return reactions, lines


def species_from_mechanism(mechanism_lines):
    """
    Extract species names from mechanism strings. Supports coefficients like '2 A' or '2A'.
    """
    species = set()
    for ln in mechanism_lines:
        if "->" not in ln:
            continue
        left, right = ln.split("->", 1)
        for side in (left, right):
            for term in side.split("+"):
                tok = term.strip()
                if not tok:
                    continue
                # Remove leading stoichiometric coefficients (e.g., '2 A' or '2A')
                tok = re.sub(r"^\s*(\d+(\.\d+)?)\s*", "", tok)
                # If something like '2A' remains merged, split digits from letters at the start
                m = re.match(r"^(\d+(\.\d+)?)([A-Za-z_].*)$", tok)
                if m:
                    tok = m.group(3).strip()
                if tok and tok.lower() != "time":
                    species.add(tok)
    return sorted(species)


def simulate_dynamic(mechanism_lines, k_fwd, k_rev, c_dict, t_stop, num_points, selections=None):
    """
    mechanism_lines: list[str] e.g. ['A -> B', 'B -> C']
    k_fwd: dict[int,float] mapping reaction id -> forward rate (k{id})
    k_rev: dict[int,float] mapping reaction id -> reverse rate (kN{id})
    c_dict: dict[str,float] initial concentrations for species
    """
    reactions = [TEReaction(reaction_string=ln, id_=i + 1) for i, ln in enumerate(mechanism_lines)]

    k_dict = {}
    for idx in range(1, len(mechanism_lines) + 1):
        k_dict[f"k{idx}"] = float(k_fwd.get(idx, 0.0))
        k_dict[f"kN{idx}"] = float(k_rev.get(idx, 0.0))

    sim = Simulator()
    sim.setup(reactions=reactions, k_dict=k_dict, c_dict=c_dict)

    sel = list(dict.fromkeys(selections or []))
    if "time" not in sel:
        sel = ["time"] + sel

    sim.simulate(0.0, float(t_stop), int(num_points), use_const_cat=False, selections=sel)

    df = sim.result.copy()
    ordered_cols = [c for c in sel if c in df.columns] + [c for c in df.columns if c not in sel]
    df = df[ordered_cols]
    return df


def make_figure(df, species_to_plot, yscale="linear", title_suffix=""):
    fig = go.Figure()

    palette = [
        "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b",
        "#7f7f7f", "#17becf", "#bcbd22", "#ff7f0e", "#e377c2",
    ]
    color_map = {}

    if "time" not in df.columns:
        return go.Figure().add_annotation(text="No 'time' column in results", showarrow=False)

    for i, sp in enumerate(species_to_plot or []):
        if sp not in df.columns:
            continue
        color_map.setdefault(sp, palette[i % len(palette)])
        fig.add_trace(
            go.Scatter(
                x=df["time"], y=df[sp],
                mode="lines",
                line=dict(width=2, color=color_map[sp]),
                name=sp,
                hovertemplate=f"time=%{{x:.4g}}<br>{sp}=%{{y:.6g}}<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"Concentration vs Time{title_suffix}",
        xaxis_title="Time",
        yaxis_title="Concentration",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(title="Species", bgcolor="rgba(255,255,255,0.6)"),
        xaxis=dict(rangeslider=dict(visible=True), zeroline=False, showgrid=True),
        yaxis=dict(zeroline=False, showgrid=True, type=yscale),
        margin=dict(l=60, r=20, t=60, b=60),
        uirevision="keep",  # preserve zoom between runs
    )
    return fig


def _default_ic_for_species(sp):
    # Heuristics for initial guesses
    if sp == "A":
        return 0.10
    if sp.lower().startswith("cat"):
        return 0.003 if sp == "cat" else 0.0
    if sp == "P":
        return 0.0
    return 0.0


box = {"border": "1px solid #eee", "borderRadius": "8px", "padding": "12px"}

app = Dash(__name__)
server = app.server

app.layout = html.Div(
    style={"fontFamily": "Inter, Arial, sans-serif", "maxWidth": "1100px", "margin": "0 auto", "padding": "16px"},
    children=[
        dcc.Store(id="mechanism-store"),
        html.H2("General Reaction Mechanism Simulator (DataTable controls)"),
        html.P("Type your mechanism below (one elementary step per line). Use '->' for direction; reverse is controlled by k_reverse in the table."),
        dcc.Textarea(
            id="mech-text",
            value=DEFAULT_MECHANISM,
            style={"width": "100%", "height": "120px", "fontFamily": "monospace"},
        ),

        html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(2, 1fr)", "gap": "16px", "marginTop": "12px"}, children=[
            html.Div(style=box, children=[
                html.H4("Rate constants"),
                dash_table.DataTable(
                    id="rates-table",
                    columns=[
                        {"name": "Step", "id": "step", "type": "numeric", "editable": False},
                        {"name": "Reaction", "id": "reaction", "type": "text", "editable": False},
                        {"name": "k_forward", "id": "k_forward", "type": "numeric"},
                        {"name": "k_reverse", "id": "k_reverse", "type": "numeric"},
                    ],
                    data=[],
                    editable=True,
                    cell_selectable=True,
                    style_table={"overflowX": "auto", "maxHeight": "360px", "overflowY": "auto"},
                    style_cell={"fontFamily": "monospace", "fontSize": "13px", "padding": "6px"},
                    style_header={"fontWeight": "600"},
                ),
                html.Small("Tip: Edit cells directly. Use arrow keys, copy/paste, and multi-cell fill."),
            ]),
            html.Div(style=box, children=[
                html.H4("Initial concentrations"),
                dash_table.DataTable(
                    id="ic-table",
                    columns=[
                        {"name": "Species", "id": "species", "type": "text", "editable": False},
                        {"name": "c0", "id": "c0", "type": "numeric"},
                    ],
                    data=[],
                    editable=True,
                    cell_selectable=True,
                    style_table={"overflowX": "auto", "maxHeight": "360px", "overflowY": "auto"},
                    style_cell={"fontFamily": "monospace", "fontSize": "13px", "padding": "6px"},
                    style_header={"fontWeight": "600"},
                ),
            ]),
        ]),

        html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(2, 1fr)", "gap": "16px", "marginTop": "16px"}, children=[
            html.Div(style=box, children=[
                html.H4("Plot Options"),
                html.Label("Species to Plot"),
                dcc.Checklist(id="species", options=[], value=[], inline=True, inputStyle={"marginRight": "6px"}),
                html.Label("Y-axis scale"),
                dcc.RadioItems(
                    id="yscale",
                    options=[{"label": "Linear", "value": "linear"}, {"label": "Log", "value": "log"}],
                    value="linear", inline=True
                ),
            ]),
            html.Div(style=box, children=[
                html.H4("Time Window"),
                html.Label("t_stop"),
                dcc.Input(id="t-stop", type="number", value=50.0, min=0.001, step=1, debounce=True, style={"width": "100%"}),
                html.Label("num_points"),
                dcc.Input(id="num-points", type="number", value=300, min=25, step=25, debounce=True, style={"width": "100%"}),
                html.Button("Run simulation", id="run", n_clicks=0, style={"marginTop": "10px"}),
            ]),
        ]),

        dcc.Graph(id="conc-graph", config={"displaylogo": False}, style={"marginTop": "16px"}),

        html.Div(style=box, children=[
            html.H4("Run Metrics"),
            html.Div(id="metrics-readout", style={"whiteSpace": "pre-wrap", "fontFamily": "monospace"}),
        ]),
    ]
)


@app.callback(
    Output("rates-table", "data"),
    Output("ic-table", "data"),
    Output("species", "options"),
    Output("species", "value"),
    Output("mechanism-store", "data"),
    Input("mech-text", "value"),
)
def on_mechanism_change(mech_text):
    try:
        reactions, lines = parse_mechanism_text(mech_text)
        sp_list = species_from_mechanism(lines)

        rates_rows = [
            {"step": i, "reaction": ln, "k_forward": K_FWD_DEFAULT, "k_reverse": K_REV_DEFAULT}
            for i, ln in enumerate(lines, start=1)
        ]
        ic_rows = [{"species": sp, "c0": _default_ic_for_species(sp)} for sp in sp_list]

        # Default plotted species: if A and P exist, plot them; else plot up to first 2
        options = [{"label": s, "value": s} for s in sp_list]
        if "A" in sp_list and "P" in sp_list:
            value = ["A", "P"]
        else:
            value = sp_list[:2]

        store = {"lines": lines, "species": sp_list}
        return rates_rows, ic_rows, options, value, store
    except Exception as e:
        # In case of parse error, clear tables and options
        return [], [], [], [], None


@app.callback(
    Output("conc-graph", "figure"),
    Output("metrics-readout", "children"),
    # Only trigger on Run button or when mechanism changes (which also repopulates tables)
    Input("run", "n_clicks"),
    Input("mechanism-store", "data"),
    # Everything else is State so it doesn't trigger compute while changing
    State("rates-table", "data"),
    State("ic-table", "data"),
    State("t-stop", "value"),
    State("num-points", "value"),
    State("yscale", "value"),
    State("species", "value"),
)
def update_plot(n_clicks, mech_store, rates_rows, ic_rows, t_stop, num_points, yscale, species_to_plot):
    # Parse mechanism from store
    try:
        if not mech_store:
            raise ValueError("No mechanism available. Please enter a valid mechanism.")
        lines = mech_store.get("lines") or []
        sp_list = mech_store.get("species") or []
        if not lines or not sp_list:
            raise ValueError("Mechanism is empty after parsing.")
    except Exception as e:
        fig = go.Figure().add_annotation(text=f"Mechanism error: {e}", showarrow=False, font=dict(color="crimson", size=14))
        fig.update_layout(template="plotly_white")
        return fig, f"Mechanism error: {e}"

    # Guardrails
    try:
        t_stop = max(1e-6, float(t_stop or 50.0))
        num_points = max(25, int(num_points or 300))
        yscale = yscale or "linear"
        species_to_plot = species_to_plot or (["A", "P"] if {"A", "P"} <= set(sp_list) else sp_list[:2])
    except Exception:
        t_stop, num_points, yscale = 50.0, 300, "linear"

    # Map k values from the rates table
    k_fwd, k_rev = {}, {}
    try:
        for row in rates_rows or []:
            idx = int(row.get("step", 0))
            kf = row.get("k_forward", 0.0)
            kr = row.get("k_reverse", 0.0)
            # Coerce and clamp
            k_fwd[idx] = max(0.0, float(kf if kf is not None else 0.0))
            k_rev[idx] = max(0.0, float(kr if kr is not None else 0.0))
    except Exception:
        pass

    # Map initial concentrations from the IC table
    c_dict = {sp: 0.0 for sp in sp_list}
    try:
        for row in ic_rows or []:
            sp = row.get("species")
            c0 = row.get("c0", 0.0)
            if sp in c_dict:
                c_dict[sp] = max(0.0, float(c0 if c0 is not None else 0.0))
    except Exception:
        pass

    # Run simulation
    try:
        df = simulate_dynamic(
            lines, k_fwd=k_fwd, k_rev=k_rev, c_dict=c_dict,
            t_stop=t_stop, num_points=num_points, selections=["time"] + sp_list
        )
        title_suffix = f" | t_stop={t_stop:g}"
        fig = make_figure(df, species_to_plot=species_to_plot, yscale=yscale, title_suffix=title_suffix)

        # Build metrics
        metrics = []
        if "A" in df.columns and "P" in df.columns:
            A0 = c_dict.get("A", 0.0)
            P_final = float(df["P"].iloc[-1])
            conv = (P_final / A0) if A0 > 0 else 0.0
            metrics.append(f"Final conversion (P/A0): {conv*100:.2f}%")

        # Always show final concentration snapshot
        last_row = df.iloc[-1]
        for sp in sp_list[:12]:  # cap to avoid overly long output
            metrics.append(f"{sp}: {float(last_row.get(sp, 0.0)):.6g}")

        # Rate summary
        rate_lines = []
        nsteps = len(lines)
        for i in range(1, nsteps + 1):
            rate_lines.append(f"Step {i}: k{i}={k_fwd.get(i, 0.0):.6g}, kN{i}={k_rev.get(i, 0.0):.6g}  |  {lines[i-1]}")
        metrics.append("\n".join(rate_lines))

        metrics_txt = "\n".join(metrics)
        return fig, metrics_txt
    except Exception as e:
        fig = go.Figure().add_annotation(text=f"Simulation error: {e}", showarrow=False, font=dict(color="crimson", size=14))
        fig.update_layout(template="plotly_white")
        return fig, f"Simulation error: {e}"


if __name__ == "__main__":
    # Dash >= 2.18: app.run
    app.run(debug=True)