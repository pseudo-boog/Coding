"""
Monte Carlo simulation: probability of winning Kings Corner on the first turn (4 players).
"""

import random
import time
import sys
sys.setrecursionlimit(10000)

NUM_SIMULATIONS = 1_000_000


def solve(hand, ftops, fbots, ctops, depth=0):
    if not hand:
        return True
    if depth > 80:
        return False
    
    # Kings need empty corners
    kings = sum(1 for r, c in hand if r == 13)
    ec = sum(1 for x in ctops if x is None)
    if kings > ec:
        return False
    
    # Play kings first (forced)
    for idx in range(len(hand)):
        r, c = hand[idx]
        if r == 13:
            rest = hand[:idx] + hand[idx+1:]
            for ci in range(4):
                if ctops[ci] is None:
                    ctops[ci] = (r, c)
                    result = solve(rest, ftops, fbots, ctops, depth+1)
                    ctops[ci] = None
                    return result
            return False
    
    # Find the most constrained card that has at least 1 option
    best_idx = -1
    best_options = None
    best_count = 999
    all_stuck = True
    
    for idx in range(len(hand)):
        r, c = hand[idx]
        options = []
        
        for fi in range(4):
            if ftops[fi] is not None:
                fr, fc = ftops[fi]
                if c != fc and r == fr - 1:
                    options.append(('f', fi))
        
        for ci in range(4):
            if ctops[ci] is not None:
                cr, cc = ctops[ci]
                if c != cc and r == cr - 1:
                    options.append(('c', ci))
        
        for fi in range(4):
            if ftops[fi] is None:
                options.append(('ef', fi))
                break
        
        if options:
            all_stuck = False
            if len(options) < best_count:
                best_count = len(options)
                best_idx = idx
                best_options = options
    
    # Try playing the most constrained card
    if not all_stuck:
        r, c = hand[best_idx]
        rest = hand[:best_idx] + hand[best_idx+1:]
        card = (r, c)
        
        for opt_type, opt_idx in best_options:
            if opt_type == 'f':
                old = ftops[opt_idx]
                ftops[opt_idx] = card
                if solve(rest, ftops, fbots, ctops, depth+1):
                    return True
                ftops[opt_idx] = old
            elif opt_type == 'c':
                old = ctops[opt_idx]
                ctops[opt_idx] = card
                if solve(rest, ftops, fbots, ctops, depth+1):
                    return True
                ctops[opt_idx] = old
            elif opt_type == 'ef':
                fi = opt_idx
                ftops[fi] = card
                fbots[fi] = card
                if solve(rest, ftops, fbots, ctops, depth+1):
                    return True
                ftops[fi] = None
                fbots[fi] = None
    
    # Try pile moves
    for fi in range(4):
        if fbots[fi] is None:
            continue
        br, bc = fbots[fi]
        
        for fi2 in range(4):
            if fi2 == fi or ftops[fi2] is None:
                continue
            fr2, fc2 = ftops[fi2]
            if bc != fc2 and br == fr2 - 1:
                old_ft2 = ftops[fi2]
                old_ft = ftops[fi]
                old_fb = fbots[fi]
                ftops[fi2] = old_ft
                ftops[fi] = None
                fbots[fi] = None
                if solve(hand, ftops, fbots, ctops, depth+1):
                    return True
                ftops[fi2] = old_ft2
                ftops[fi] = old_ft
                fbots[fi] = old_fb
        
        for ci in range(4):
            if ctops[ci] is None:
                continue
            cr, cc = ctops[ci]
            if bc != cc and br == cr - 1:
                old_ct = ctops[ci]
                old_ft = ftops[fi]
                old_fb = fbots[fi]
                ctops[ci] = old_ft
                ftops[fi] = None
                fbots[fi] = None
                if solve(hand, ftops, fbots, ctops, depth+1):
                    return True
                ctops[ci] = old_ct
                ftops[fi] = old_ft
                fbots[fi] = old_fb
    
    return False


def simulate_one(deck):
    random.shuffle(deck)
    hand = list(deck[:7]) + [deck[32]]
    founds = deck[28:32]
    ftops = list(founds)
    fbots = list(founds)
    ctops = [None, None, None, None]
    return solve(hand, ftops, fbots, ctops)


def main():
    deck = []
    for color in (0, 0, 1, 1):
        for rank in range(1, 14):
            deck.append((rank, color))
    
    wins = 0
    start = time.time()
    report_every = max(1, NUM_SIMULATIONS // 20)
    
    print(f"Running {NUM_SIMULATIONS:,} Kings Corner simulations (4 players, first turn)...", flush=True)
    print(flush=True)
    
    for i in range(1, NUM_SIMULATIONS + 1):
        if simulate_one(deck):
            wins += 1
        if i % report_every == 0:
            elapsed = time.time() - start
            rate = i / elapsed
            pct = wins / i * 100
            eta = (NUM_SIMULATIONS - i) / rate
            print(f"  {i:>10,} / {NUM_SIMULATIONS:,}  |  Wins: {wins:,} ({pct:.4f}%)  |  {rate:.0f}/s  |  ETA: {eta:.0f}s", flush=True)
    
    elapsed = time.time() - start
    pct = wins / NUM_SIMULATIONS * 100
    
    print(flush=True)
    print(f"{'='*55}", flush=True)
    print(f"  Total simulations:  {NUM_SIMULATIONS:,}", flush=True)
    print(f"  First-turn wins:    {wins:,}", flush=True)
    print(f"  Win probability:    {pct:.4f}%", flush=True)
    if wins > 0:
        print(f"  Odds:               ~1 in {NUM_SIMULATIONS // wins:,}", flush=True)
    else:
        print(f"  Odds:               0 wins detected", flush=True)
    print(f"  Time elapsed:       {elapsed:.1f}s", flush=True)
    print(f"{'='*55}", flush=True)


if __name__ == "__main__":
    main()
