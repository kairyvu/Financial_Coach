"""
Run full evaluation across all personas.

This script:
1. Generates synthetic data for each persona
2. Trains AE models
3. Runs evaluation comparing AE vs rule baseline
4. Produces metrics report (CSV + markdown)
"""

import os
import sys
from datetime import datetime, timedelta

import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.generator import generate_persona_dataset
from src.data.personas import list_personas
from src.mcp_server.client import AccountantClient
from src.host.controller import HostController
from src.evaluation.metrics import Evaluator


def run_full_evaluation(
    days_train: int = 60,
    days_test: int = 30,
    epochs: int = 30,
    output_dir: str = "results",
    verbose: bool = True,
):
    """
    Run complete evaluation across all personas.
    
    Args:
        days_train: Days of data for training
        days_test: Days of data for testing
        epochs: Training epochs for AE
        output_dir: Directory for output files
        verbose: Print progress
    """
    print("=" * 60)
    print("Financial Coach - Full Evaluation")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Training days: {days_train}")
    print(f"  Test days: {days_test}")
    print(f"  AE epochs: {epochs}")
    print(f"  Output: {output_dir}/")
    
    evaluator = Evaluator()
    accountant = AccountantClient()
    
    personas = list_personas()
    print(f"\nEvaluating {len(personas)} personas: {', '.join(personas)}")
    
    for i, persona_name in enumerate(personas):
        print(f"\n[{i+1}/{len(personas)}] Processing {persona_name}...")
        
        # Generate training data
        base_seed = 42 + i * 100
        train_df = generate_persona_dataset(
            persona_name=persona_name,
            days=days_train,
            seed=base_seed,
        )
        
        # Generate test data (different seed, later time period)
        test_df = generate_persona_dataset(
            persona_name=persona_name,
            days=days_test,
            seed=base_seed + 1000,
            start_date=datetime.now() - timedelta(days=days_test),
        )
        
        if verbose:
            print(f"  Training data: {len(train_df)} transactions")
            print(f"  Test data: {len(test_df)} transactions")
            print(f"  Ground truth anomalies in test: {test_df['is_anomaly'].sum()}")
        
        # Train model
        print(f"  Training AE model...")
        result = accountant.train_persona_model(
            persona_name=persona_name,
            days=days_train,
            epochs=epochs,
            seed=base_seed,
            verbose=False,
        )
        
        if verbose:
            print(f"  Model trained: threshold={result.threshold:.6f}")
        
        # Create controller and evaluate
        controller = HostController(
            persona_name=persona_name,
            accountant=accountant,
        )
        
        eval_result = evaluator.evaluate_persona(
            controller=controller,
            test_df=test_df,
            training_df=train_df,
            verbose=verbose,
        )
    
    # Save results
    print("\n" + "=" * 60)
    print("Saving Results")
    print("=" * 60)
    
    evaluator.save_results(output_dir)
    
    # Print final summary
    print("\n" + "=" * 60)
    print("Final Summary")
    print("=" * 60)
    
    summary_df = evaluator.get_summary_df()
    
    print(f"\n📊 Average Metrics Across All Personas:")
    print(f"   Precision:      {summary_df['precision'].mean():.3f}")
    print(f"   Recall:         {summary_df['recall'].mean():.3f}")
    print(f"   F1 Score:       {summary_df['f1_score'].mean():.3f}")
    print(f"   Nudges/Week:    {summary_df['nudges_per_week'].mean():.1f}")
    print(f"   Savings Uplift: {summary_df['savings_uplift'].mean():+.1%} vs rule baseline")
    print(f"   Avg Latency:    {summary_df['avg_latency_ms'].mean():.2f}ms")
    
    print(f"\n✅ Evaluation complete! Results saved to {output_dir}/")
    
    return evaluator


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Financial Coach evaluation")
    parser.add_argument("--train-days", type=int, default=60, help="Days of training data")
    parser.add_argument("--test-days", type=int, default=30, help="Days of test data")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--quiet", action="store_true", help="Reduce output")
    
    args = parser.parse_args()
    
    run_full_evaluation(
        days_train=args.train_days,
        days_test=args.test_days,
        epochs=args.epochs,
        output_dir=args.output,
        verbose=not args.quiet,
    )

