#!/usr/bin/env python3
"""
Financial Coach Demo Script

Demonstrates the end-to-end sense-and-respond loop:
1. Generate synthetic transaction stream
2. Train persona-specific autoencoder
3. Replay transactions with real-time nudges
4. Show summary metrics

Usage:
    python demo.py                          # Run demo with default persona
    python demo.py --persona spender_sarah  # Run with specific persona
    python demo.py --all                    # Run demo for all personas
"""

import os
import sys
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.generator import generate_persona_dataset
from src.data.personas import list_personas, get_persona
from src.mcp_server.client import AccountantClient
from src.host.controller import HostController, create_controller
from src.host.policy import Action


def print_header(text: str, char: str = "="):
    """Print a formatted header."""
    width = 60
    print()
    print(char * width)
    print(f" {text}")
    print(char * width)


def print_persona_info(persona_name: str):
    """Print persona details."""
    persona = get_persona(persona_name)
    print(f"\n👤 Persona: {persona_name}")
    print(f"   Monthly Income:  ${persona.monthly_income:,.0f}")
    print(f"   Savings Goal:    ${persona.savings_goal:,.0f} ({100*persona.savings_goal/persona.monthly_income:.0f}%)")
    print(f"   Monthly Budget:  ${persona.monthly_income - persona.savings_goal:,.0f}")
    print(f"   Spending Style:  {'High' if persona.impulse_probability > 0.15 else 'Moderate' if persona.impulse_probability > 0.08 else 'Conservative'}")


def run_demo(
    persona_name: str = "balanced_ben",
    days: int = 30,
    verbose: bool = True,
):
    """
    Run the Financial Coach demo for a single persona.
    
    Args:
        persona_name: Name of the persona to demo
        days: Days of transaction data to process
        verbose: Print detailed output
    """
    print_header(f"Financial Coach Demo - {persona_name}")
    print_persona_info(persona_name)
    
    # Step 1: Generate synthetic data
    print_header("Step 1: Generating Transaction Data", "-")
    
    df = generate_persona_dataset(
        persona_name=persona_name,
        days=days,
        seed=42,
    )
    
    print(f"Generated {len(df)} transactions over {days} days")
    print(f"Ground truth anomalies: {df['is_anomaly'].sum()} ({100*df['is_anomaly'].mean():.1f}%)")
    
    # Show sample transactions
    print("\n📋 Sample transactions:")
    sample = df.head(5)[["timestamp", "amount", "category", "is_anomaly"]]
    for _, row in sample.iterrows():
        anomaly_marker = "⚠️" if row["is_anomaly"] else "  "
        print(f"   {anomaly_marker} {row['timestamp'].strftime('%Y-%m-%d %H:%M')} | "
              f"${row['amount']:>8.2f} | {row['category']}")
    
    # Step 2: Train the model
    print_header("Step 2: Training Autoencoder Model", "-")
    
    accountant = AccountantClient()
    result = accountant.train_persona_model(
        persona_name=persona_name,
        days=60,  # Train on more data
        epochs=30,
        seed=42,
        verbose=True,
    )
    
    print(f"\n✅ Model trained successfully!")
    print(f"   Threshold: {result.threshold:.6f}")
    print(f"   Epochs: {result.epochs_trained}")
    
    # Step 3: Replay transactions with nudges
    print_header("Step 3: Processing Transactions (Sense & Respond)", "-")
    
    controller = create_controller(persona_name, train_if_needed=False)
    
    nudge_count = 0
    anomaly_count = 0
    
    print("\n🔄 Processing transaction stream...")
    print("   (Showing nudges only)\n")
    
    for decision in controller.replay_transactions(df, verbose=False):
        if decision.scoring_result.is_anomaly:
            anomaly_count += 1
        
        if decision.action != Action.DO_NOTHING:
            nudge_count += 1
            if verbose and nudge_count <= 10:  # Show first 10 nudges
                timestamp = decision.timestamp.strftime('%Y-%m-%d %H:%M')
                action_icon = "🔔" if decision.action == Action.GENTLE_NUDGE else "🚨"
                print(f"   {action_icon} [{timestamp}]")
                print(f"      {decision.message}")
                print()
    
    if nudge_count > 10:
        print(f"   ... and {nudge_count - 10} more nudges")
    
    # Step 4: Show summary
    print_header("Step 4: Session Summary", "-")
    
    summary = controller.get_summary()
    
    print(f"\n📊 Results:")
    print(f"   Transactions Processed: {summary['transactions_processed']}")
    print(f"   Anomalies Detected:     {summary['anomalies_detected']}")
    print(f"   Total Nudges Sent:      {summary['nudges_sent']}")
    
    if summary['action_counts']:
        print(f"\n   Action Breakdown:")
        for action, count in summary['action_counts'].items():
            print(f"      {action}: {count}")
    
    print(f"\n💰 Budget Status:")
    print(f"   Spent This Month:  ${summary['spent_current_month']:,.2f}")
    print(f"   Monthly Budget:    ${summary['budget']:,.2f}")
    print(f"   Savings Goal:      ${summary['savings_goal']:,.2f}")
    
    # Calculate detection accuracy
    decisions_df = controller.get_decisions_df()
    if not decisions_df.empty and "is_anomaly_predicted" in decisions_df.columns:
        ground_truth = df["is_anomaly"].values[:len(decisions_df)]
        predicted = decisions_df["is_anomaly_predicted"].values
        
        if len(ground_truth) == len(predicted):
            tp = ((predicted == True) & (ground_truth == True)).sum()
            fp = ((predicted == True) & (ground_truth == False)).sum()
            fn = ((predicted == False) & (ground_truth == True)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            print(f"\n🎯 Detection Accuracy:")
            print(f"   Precision: {precision:.1%}")
            print(f"   Recall:    {recall:.1%}")
    
    print("\n" + "=" * 60)
    print(" Demo Complete!")
    print("=" * 60)
    
    return controller


def run_all_personas_demo():
    """Run demo for all personas with summary."""
    print_header("Financial Coach - All Personas Demo")
    
    personas = list_personas()
    print(f"\nRunning demo for {len(personas)} personas...")
    
    results = {}
    
    for persona_name in personas:
        try:
            controller = run_demo(persona_name, days=30, verbose=False)
            results[persona_name] = controller.get_summary()
        except Exception as e:
            print(f"Error with {persona_name}: {e}")
            results[persona_name] = None
    
    # Print comparison table
    print_header("Comparison Summary")
    
    print("\n| Persona | Transactions | Anomalies | Nudges | Budget Status |")
    print("|---------|--------------|-----------|--------|---------------|")
    
    for persona_name, summary in results.items():
        if summary:
            budget_pct = summary['spent_current_month'] / summary['budget'] * 100
            status = "✅ On Track" if budget_pct < 100 else "⚠️ Over"
            print(f"| {persona_name:15} | {summary['transactions_processed']:12} | "
                  f"{summary['anomalies_detected']:9} | {summary['nudges_sent']:6} | {status:13} |")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Financial Coach Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python demo.py                          # Demo with balanced_ben
    python demo.py --persona spender_sarah  # Demo with specific persona
    python demo.py --all                    # Demo all personas
    python demo.py --days 60                # Process 60 days of data
        """
    )
    
    parser.add_argument(
        "--persona", 
        type=str, 
        default="balanced_ben",
        choices=list_personas(),
        help="Persona to demo"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Days of transaction data"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run demo for all personas"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )
    
    args = parser.parse_args()
    
    if args.all:
        run_all_personas_demo()
    else:
        run_demo(
            persona_name=args.persona,
            days=args.days,
            verbose=not args.quiet,
        )

