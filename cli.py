#!/usr/bin/env python3
"""
Financial Coach CLI

Main command-line interface for the Financial Coach system.

Commands:
    train     - Train a persona model
    score     - Score a single transaction
    replay    - Replay transactions from a file
    evaluate  - Run full evaluation
    demo      - Run interactive demo
"""

import os
import sys
import json
from datetime import datetime

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.personas import list_personas, get_persona
from src.data.generator import generate_persona_dataset
from src.mcp_server.client import AccountantClient
from src.host.controller import HostController, create_controller

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Financial Coach - AI-powered savings nudge system."""
    pass


@cli.command()
@click.argument("persona", type=click.Choice(list_personas()))
@click.option("--days", default=60, help="Days of training data")
@click.option("--epochs", default=30, help="Training epochs")
@click.option("--seed", default=42, help="Random seed")
def train(persona: str, days: int, epochs: int, seed: int):
    """Train an autoencoder model for a persona."""
    console.print(f"\n[bold blue]Training model for {persona}[/bold blue]")
    
    client = AccountantClient()
    
    with console.status("Training..."):
        result = client.train_persona_model(
            persona_name=persona,
            days=days,
            epochs=epochs,
            seed=seed,
            verbose=False,
        )
    
    console.print(Panel(
        f"[green]✓ Training complete![/green]\n\n"
        f"Transactions: {result.transactions_used}\n"
        f"Epochs: {result.epochs_trained}\n"
        f"Threshold: {result.threshold:.6f}\n"
        f"Model saved to: {result.model_dir}",
        title="Training Result"
    ))


@cli.command()
@click.argument("persona", type=click.Choice(list_personas()))
@click.option("--amount", required=True, type=float, help="Transaction amount")
@click.option("--category", required=True, 
              type=click.Choice(["groceries", "utilities", "transport", "dining",
                                "entertainment", "shopping", "healthcare", 
                                "subscriptions", "misc"]),
              help="Spending category")
@click.option("--hour", default=12, type=int, help="Hour of transaction (0-23)")
@click.option("--day", default=2, type=int, help="Day of week (0=Mon, 6=Sun)")
@click.option("--days-to-end", default=15, type=int, help="Days to month end")
def score(persona: str, amount: float, category: str, hour: int, day: int, days_to_end: int):
    """Score a single transaction for anomaly detection."""
    client = AccountantClient()
    
    # Check if model exists
    status = client.get_model_status(persona)
    if status["status"] != "ready":
        console.print(f"[red]No model found for {persona}. Run 'train' first.[/red]")
        return
    
    result = client.score_transaction(
        persona_name=persona,
        amount=amount,
        category=category,
        hour=hour,
        day_of_week=day,
        days_to_month_end=days_to_end,
    )
    
    anomaly_status = "[red]⚠️ ANOMALY[/red]" if result.is_anomaly else "[green]✓ Normal[/green]"
    
    console.print(Panel(
        f"Transaction: ${amount:.2f} on {category}\n"
        f"Status: {anomaly_status}\n\n"
        f"Reconstruction Error: {result.recon_error:.6f}\n"
        f"Threshold: {result.threshold:.6f}\n"
        f"Error Ratio: {result.error_ratio:.2f}x",
        title=f"Scoring Result - {persona}"
    ))


@cli.command()
@click.argument("persona", type=click.Choice(list_personas()))
@click.option("--days", default=30, help="Days of transactions to replay")
@click.option("--seed", default=100, help="Random seed for data generation")
@click.option("--output", default=None, help="Save decisions to file")
def replay(persona: str, days: int, seed: int, output: str):
    """Replay a synthetic transaction stream with nudges."""
    console.print(f"\n[bold blue]Replaying {days} days for {persona}[/bold blue]")
    
    # Generate data
    df = generate_persona_dataset(persona, days=days, seed=seed)
    console.print(f"Generated {len(df)} transactions")
    
    # Create controller
    controller = create_controller(persona, train_if_needed=True)
    
    # Replay
    nudge_count = 0
    for decision in controller.replay_transactions(df, verbose=False):
        if decision.message:
            nudge_count += 1
            timestamp = decision.timestamp.strftime('%Y-%m-%d %H:%M')
            console.print(f"\n[yellow][{timestamp}][/yellow]")
            console.print(f"  {decision.message}")
    
    # Summary
    summary = controller.get_summary()
    
    table = Table(title="Session Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Transactions", str(summary["transactions_processed"]))
    table.add_row("Anomalies Detected", str(summary["anomalies_detected"]))
    table.add_row("Nudges Sent", str(summary["nudges_sent"]))
    table.add_row("Spent This Month", f"${summary['spent_current_month']:,.2f}")
    table.add_row("Budget", f"${summary['budget']:,.2f}")
    
    console.print("\n")
    console.print(table)
    
    if output:
        controller.save_log(output)
        console.print(f"\n[green]Saved log to {output}[/green]")


@cli.command()
@click.option("--train-days", default=60, help="Days of training data")
@click.option("--test-days", default=30, help="Days of test data")
@click.option("--epochs", default=30, help="Training epochs")
@click.option("--output", default="results", help="Output directory")
def evaluate(train_days: int, test_days: int, epochs: int, output: str):
    """Run full evaluation across all personas."""
    console.print("\n[bold blue]Running Full Evaluation[/bold blue]")
    
    from src.evaluation.run_evaluation import run_full_evaluation
    
    run_full_evaluation(
        days_train=train_days,
        days_test=test_days,
        epochs=epochs,
        output_dir=output,
        verbose=True,
    )


@cli.command()
def personas():
    """List available personas."""
    table = Table(title="Available Personas")
    table.add_column("Name", style="cyan")
    table.add_column("Income", style="green")
    table.add_column("Savings Goal", style="yellow")
    table.add_column("Style", style="magenta")
    
    for name in list_personas():
        persona = get_persona(name)
        savings_pct = persona.savings_goal / persona.monthly_income * 100
        style = "High" if persona.impulse_probability > 0.15 else \
                "Moderate" if persona.impulse_probability > 0.08 else "Conservative"
        
        table.add_row(
            name,
            f"${persona.monthly_income:,.0f}",
            f"${persona.savings_goal:,.0f} ({savings_pct:.0f}%)",
            style
        )
    
    console.print(table)


@cli.command()
def status():
    """Check status of trained models."""
    client = AccountantClient()
    
    table = Table(title="Model Status")
    table.add_column("Persona", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Threshold", style="yellow")
    
    for name in list_personas():
        status = client.get_model_status(name)
        if status["status"] == "ready":
            table.add_row(name, "✓ Ready", f"{status['threshold']:.6f}")
        else:
            table.add_row(name, "[red]✗ Not trained[/red]", "-")
    
    console.print(table)


if __name__ == "__main__":
    cli()

