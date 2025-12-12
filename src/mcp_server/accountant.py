"""
MCP Accountant Server - Exposes AE training and scoring tools.

This server provides two main tools:
1. train_persona_model - Train a persona-specific autoencoder
2. score_transaction - Score a transaction for anomaly detection
"""

import json
import os
import sys
from datetime import datetime
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.autoencoder import AnomalyDetector
from src.data.generator import generate_persona_dataset
from src.data.personas import list_personas


# Default artifacts directory
ARTIFACTS_DIR = os.environ.get("FINANCIAL_COACH_ARTIFACTS", "artifacts")


def get_model_dir(persona_name: str) -> str:
    """Get the model directory for a persona."""
    return os.path.join(ARTIFACTS_DIR, persona_name)


# Create the MCP server
server = Server("accountant")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="train_persona_model",
            description="""Train a persona-specific autoencoder model for anomaly detection.
            
This tool:
1. Generates or loads transaction data for the specified persona
2. Trains an autoencoder to learn normal spending patterns
3. Computes anomaly threshold (95th percentile of reconstruction errors)
4. Saves all artifacts for future scoring

Parameters:
- persona_name: One of the available personas (frugal_frank, spender_sarah, balanced_ben, student_sam, executive_emma)
- days: Number of days of transaction history to train on (default: 60)
- epochs: Maximum training epochs (default: 30)
- seed: Random seed for reproducibility (default: 42)
""",
            inputSchema={
                "type": "object",
                "properties": {
                    "persona_name": {
                        "type": "string",
                        "description": "Name of the persona to train model for",
                        "enum": list_personas(),
                    },
                    "days": {
                        "type": "integer",
                        "description": "Number of days of transaction history",
                        "default": 60,
                    },
                    "epochs": {
                        "type": "integer",
                        "description": "Maximum training epochs",
                        "default": 30,
                    },
                    "seed": {
                        "type": "integer",
                        "description": "Random seed for reproducibility",
                        "default": 42,
                    },
                },
                "required": ["persona_name"],
            },
        ),
        Tool(
            name="score_transaction",
            description="""Score a single transaction for anomaly detection.
            
This tool:
1. Loads the trained model for the specified persona
2. Computes reconstruction error for the transaction
3. Determines if the transaction is anomalous (error > threshold)

Parameters:
- persona_name: The persona whose model to use
- amount: Transaction amount in dollars
- category: Spending category
- hour: Hour of transaction (0-23)
- day_of_week: Day of week (0=Monday, 6=Sunday)
- days_to_month_end: Days until end of month

Returns reconstruction error, anomaly flag, and threshold.
""",
            inputSchema={
                "type": "object",
                "properties": {
                    "persona_name": {
                        "type": "string",
                        "description": "Name of the persona",
                        "enum": list_personas(),
                    },
                    "amount": {
                        "type": "number",
                        "description": "Transaction amount in dollars",
                    },
                    "category": {
                        "type": "string",
                        "description": "Spending category",
                        "enum": [
                            "groceries", "utilities", "transport", "dining",
                            "entertainment", "shopping", "healthcare", "subscriptions", "misc"
                        ],
                    },
                    "hour": {
                        "type": "integer",
                        "description": "Hour of transaction (0-23)",
                        "minimum": 0,
                        "maximum": 23,
                    },
                    "day_of_week": {
                        "type": "integer",
                        "description": "Day of week (0=Monday, 6=Sunday)",
                        "minimum": 0,
                        "maximum": 6,
                    },
                    "days_to_month_end": {
                        "type": "integer",
                        "description": "Days until end of month",
                        "minimum": 0,
                        "maximum": 31,
                    },
                },
                "required": ["persona_name", "amount", "category", "hour", "day_of_week", "days_to_month_end"],
            },
        ),
        Tool(
            name="get_model_status",
            description="Check if a trained model exists for a persona and get its metadata.",
            inputSchema={
                "type": "object",
                "properties": {
                    "persona_name": {
                        "type": "string",
                        "description": "Name of the persona",
                        "enum": list_personas(),
                    },
                },
                "required": ["persona_name"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""
    
    if name == "train_persona_model":
        return await train_persona_model(arguments)
    elif name == "score_transaction":
        return await score_transaction(arguments)
    elif name == "get_model_status":
        return await get_model_status(arguments)
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def train_persona_model(args: dict) -> list[TextContent]:
    """Train a persona-specific autoencoder model."""
    persona_name = args["persona_name"]
    days = args.get("days", 60)
    epochs = args.get("epochs", 30)
    seed = args.get("seed", 42)
    
    try:
        # Generate training data
        df = generate_persona_dataset(
            persona_name=persona_name,
            days=days,
            seed=seed,
        )
        
        # Create and train detector
        detector = AnomalyDetector(persona_name=persona_name)
        metrics = detector.train(
            df=df,
            epochs=epochs,
            verbose=False,
        )
        
        # Save model artifacts
        model_dir = get_model_dir(persona_name)
        detector.save(model_dir)
        
        # Also save the training data for reference
        data_path = os.path.join(model_dir, "training_data.csv")
        df.to_csv(data_path, index=False)
        
        result = {
            "status": "success",
            "persona_name": persona_name,
            "transactions_used": len(df),
            "epochs_trained": metrics["epochs_trained"],
            "final_train_loss": round(metrics["final_train_loss"], 6),
            "final_val_loss": round(metrics["final_val_loss"], 6),
            "threshold": round(metrics["threshold"], 6),
            "model_dir": model_dir,
            "anomalies_in_training_data": int(df["is_anomaly"].sum()),
        }
        
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
    except Exception as e:
        error_result = {
            "status": "error",
            "error": str(e),
            "persona_name": persona_name,
        }
        return [TextContent(type="text", text=json.dumps(error_result, indent=2))]


async def score_transaction(args: dict) -> list[TextContent]:
    """Score a single transaction for anomaly detection."""
    persona_name = args["persona_name"]
    
    try:
        # Load trained model
        model_dir = get_model_dir(persona_name)
        if not os.path.exists(model_dir):
            return [TextContent(
                type="text",
                text=json.dumps({
                    "status": "error",
                    "error": f"No trained model found for {persona_name}. Run train_persona_model first.",
                }, indent=2)
            )]
        
        detector = AnomalyDetector.load(model_dir)
        
        # Prepare transaction data
        transaction = {
            "amount": args["amount"],
            "category": args["category"],
            "hour": args["hour"],
            "day_of_week": args["day_of_week"],
            "days_to_month_end": args["days_to_month_end"],
        }
        
        # Score transaction
        result = detector.score_single(transaction)
        
        response = {
            "status": "success",
            "persona_name": persona_name,
            "transaction": transaction,
            "recon_error": round(result["recon_error"], 6),
            "is_anomaly": result["is_anomaly"],
            "threshold": round(result["threshold"], 6),
            "error_ratio": round(result["recon_error"] / result["threshold"], 3),
        }
        
        return [TextContent(type="text", text=json.dumps(response, indent=2))]
        
    except Exception as e:
        error_result = {
            "status": "error",
            "error": str(e),
            "persona_name": persona_name,
        }
        return [TextContent(type="text", text=json.dumps(error_result, indent=2))]


async def get_model_status(args: dict) -> list[TextContent]:
    """Check model status for a persona."""
    persona_name = args["persona_name"]
    model_dir = get_model_dir(persona_name)
    
    if not os.path.exists(model_dir):
        result = {
            "status": "not_found",
            "persona_name": persona_name,
            "message": "No trained model found. Run train_persona_model first.",
        }
    else:
        # Load metadata
        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        result = {
            "status": "ready",
            "persona_name": persona_name,
            "model_dir": model_dir,
            "threshold": metadata["threshold"],
            "input_dim": metadata["input_dim"],
        }
    
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

