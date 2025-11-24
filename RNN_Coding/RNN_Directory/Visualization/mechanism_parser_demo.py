#!/usr/bin/env python3
"""
Demonstration of the mechanism parser functionality for Phase_Portraits.py

This script shows how to use the parse_mechanism() function to automatically
read and process reaction mechanisms with reversible reactions.
"""

from Phase_Portraits import parse_mechanism, print_mechanism_info


def demo_basic_mechanism():
    """Demo: Basic 3-step mechanism"""
    print("\n" + "█"*70)
    print("DEMO 1: Basic 3-Step Mechanism")
    print("█"*70)
    
    reactions = [
        "A + cat -> cat1",
        "cat1 -> cat + P",
        "cat + P -> catI"
    ]
    
    mechanism = parse_mechanism(reactions, include_reverse=True)
    print_mechanism_info(mechanism)
    
    print("Forward rate constants (k_f):")
    for key, val in mechanism['k_dict'].items():
        if '_f' in key:
            print(f"  {key}: {val}")
    
    print("\nReverse rate constants (k_r) - all default to 0:")
    for key, val in mechanism['k_dict'].items():
        if '_r' in key:
            print(f"  {key}: {val}")


def demo_complex_mechanism():
    """Demo: Complex 4-step mechanism with dual inhibition"""
    print("\n" + "█"*70)
    print("DEMO 2: Complex 4-Step Mechanism (Your Example)")
    print("█"*70)
    
    reactions = [
        "A + cat -> cat1",
        "cat1 -> cat + P",
        "cat + P -> catI",
        "cat + A -> catI"
    ]
    
    mechanism = parse_mechanism(reactions, include_reverse=True)
    print_mechanism_info(mechanism)
    
    print("Species in mechanism:")
    for species in sorted(mechanism['species']):
        print(f"  • {species}")


def demo_extended_mechanism():
    """Demo: Extended mechanism with multiple reactants"""
    print("\n" + "█"*70)
    print("DEMO 3: Extended Mechanism (Multiple Reactants)")
    print("█"*70)
    
    reactions = [
        "A + B + cat -> cat1",
        "cat1 -> P + cat",
        "P + cat -> catI",
        "A + catI -> cat"  # Regeneration
    ]
    
    mechanism = parse_mechanism(reactions, include_reverse=True)
    print_mechanism_info(mechanism)


def demo_without_reverse():
    """Demo: Mechanism without automatically generated reverse reactions"""
    print("\n" + "█"*70)
    print("DEMO 4: Mechanism WITHOUT Automatic Reverse Reactions")
    print("█"*70)
    
    reactions = [
        "A + cat -> cat1",
        "cat1 -> cat + P",
        "cat + P -> catI"
    ]
    
    mechanism = parse_mechanism(reactions, include_reverse=False)
    print_mechanism_info(mechanism)
    
    print(f"Total reactions: {mechanism['num_reactions']}")
    print(f"k_dict keys: {list(mechanism['k_dict'].keys())}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("MECHANISM PARSER DEMONSTRATION")
    print("="*70)
    
    demo_basic_mechanism()
    demo_complex_mechanism()
    demo_extended_mechanism()
    demo_without_reverse()
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    print("\nUsage in your code:")
    print("""
    from Phase_Portraits import parse_mechanism, print_mechanism_info
    
    # Define your mechanism
    reactions = [
        "A + cat -> cat1",
        "cat1 -> cat + P",
        "cat + P -> catI",
        "cat + A -> catI"
    ]
    
    # Parse it
    mechanism = parse_mechanism(reactions, include_reverse=True)
    
    # View info
    print_mechanism_info(mechanism)
    
    # Access reaction data
    print(mechanism['species'])           # All species
    print(mechanism['forward_reactions']) # Forward rxns
    print(mechanism['reverse_reactions']) # Reverse rxns
    print(mechanism['k_dict'])            # Rate constants dict
    """)
    print("="*70 + "\n")
