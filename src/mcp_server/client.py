"""
Direct client for the Accountant functionality.

This provides a Python API to the same functionality as the MCP server,
allowing direct usage without going through the MCP protocol.
"""

import json
import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.models.autoencoder import AnomalyDetector
from src.data.generator import generate_persona_dataset
from src.data.personas import list_personas, get_persona


# Default artifacts directory
ARTIFACTS_DIR = os.environ.get("FINANCIAL_COACH_ARTIFACTS", "artifacts")


def get_model_dir(persona_name: str) -> str:
    """Get the model directory for a persona."""
    return os.path.join(ARTIFACTS_DIR, persona_name)


@dataclass
class ScoringResult:
    """Result of scoring a transaction."""
    recon_error: float
    is_anomaly: bool
    threshold: float
    error_ratio: float
    
    def to_dict(self) -> dict:
        return {
            "recon_error": self.recon_error,
            "is_anomaly": self.is_anomaly,
            "threshold": self.threshold,
            "error_ratio": self.error_ratio,
        }


@dataclass
class TrainingResult:
    """Result of training a model."""
    persona_name: str
    transactions_used: int
    epochs_trained: int
    final_train_loss: float
    final_val_loss: float
    threshold: float
    model_dir: str
    
    def to_dict(self) -> dict:
        return {
            "persona_name": self.persona_name,
            "transactions_used": self.transactions_used,
            "epochs_trained": self.epochs_trained,
            "final_train_loss": self.final_train_loss,
            "final_val_loss": self.final_val_loss,
            "threshold": self.threshold,
            "model_dir": self.model_dir,
        }


class AccountantClient:
    """
    Client for the Accountant anomaly detection system.
    
    Provides direct Python access to training and scoring functionality.
    """
    
    def __init__(self, artifacts_dir: str = ARTIFACTS_DIR):
        self.artifacts_dir = artifacts_dir
        self._loaded_detectors: dict[str, AnomalyDetector] = {}
    
    def _get_model_dir(self, persona_name: str) -> str:
        """Get model directory for a persona."""
        return os.path.join(self.artifacts_dir, persona_name)
    
    def train_persona_model(
        self,
        persona_name: str,
        days: int = 60,
        epochs: int = 30,
        seed: int = 42,
        verbose: bool = True,
    ) -> TrainingResult:
        """
        Train a persona-specific autoencoder model.
        
        Args:
            persona_name: Name of the persona
            days: Days of transaction history to train on
            epochs: Maximum training epochs
            seed: Random seed for reproducibility
            verbose: Print training progress
            
        Returns:
            TrainingResult with training metrics
        """
        if persona_name not in list_personas():
            raise ValueError(f"Unknown persona: {persona_name}")
        
        # Generate training data
        if verbose:
            print(f"Generating {days} days of transaction data for {persona_name}...")
        
        df = generate_persona_dataset(
            persona_name=persona_name,
            days=days,
            seed=seed,
        )
        
        if verbose:
            print(f"Generated {len(df)} transactions")
        
        # Create and train detector
        detector = AnomalyDetector(persona_name=persona_name)
        metrics = detector.train(
            df=df,
            epochs=epochs,
            verbose=verbose,
        )
        
        # Save model artifacts
        model_dir = self._get_model_dir(persona_name)
        detector.save(model_dir)
        
        # Save training data
        data_path = os.path.join(model_dir, "training_data.csv")
        df.to_csv(data_path, index=False)
        
        # Cache the detector
        self._loaded_detectors[persona_name] = detector
        
        return TrainingResult(
            persona_name=persona_name,
            transactions_used=len(df),
            epochs_trained=metrics["epochs_trained"],
            final_train_loss=metrics["final_train_loss"],
            final_val_loss=metrics["final_val_loss"],
            threshold=metrics["threshold"],
            model_dir=model_dir,
        )
    
    def _get_detector(self, persona_name: str) -> AnomalyDetector:
        """Get or load the detector for a persona."""
        if persona_name not in self._loaded_detectors:
            model_dir = self._get_model_dir(persona_name)
            if not os.path.exists(model_dir):
                raise ValueError(
                    f"No trained model found for {persona_name}. "
                    "Run train_persona_model first."
                )
            self._loaded_detectors[persona_name] = AnomalyDetector.load(model_dir)
        
        return self._loaded_detectors[persona_name]
    
    def score_transaction(
        self,
        persona_name: str,
        amount: float,
        category: str,
        hour: int,
        day_of_week: int,
        days_to_month_end: int,
    ) -> ScoringResult:
        """
        Score a single transaction for anomaly detection.
        
        Args:
            persona_name: Name of the persona
            amount: Transaction amount in dollars
            category: Spending category
            hour: Hour of transaction (0-23)
            day_of_week: Day of week (0=Monday, 6=Sunday)
            days_to_month_end: Days until end of month
            
        Returns:
            ScoringResult with anomaly detection results
        """
        detector = self._get_detector(persona_name)
        
        transaction = {
            "amount": amount,
            "category": category,
            "hour": hour,
            "day_of_week": day_of_week,
            "days_to_month_end": days_to_month_end,
        }
        
        result = detector.score_single(transaction)
        
        return ScoringResult(
            recon_error=result["recon_error"],
            is_anomaly=result["is_anomaly"],
            threshold=result["threshold"],
            error_ratio=result["recon_error"] / result["threshold"],
        )
    
    def score_transactions(
        self,
        persona_name: str,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Score multiple transactions.
        
        Args:
            persona_name: Name of the persona
            df: DataFrame with transaction data
            
        Returns:
            DataFrame with added recon_error and is_anomaly columns
        """
        detector = self._get_detector(persona_name)
        errors, is_anomaly = detector.score(df)
        
        result_df = df.copy()
        result_df["recon_error"] = errors
        result_df["is_anomaly_predicted"] = is_anomaly
        result_df["threshold"] = detector.threshold
        
        return result_df
    
    def get_model_status(self, persona_name: str) -> dict:
        """Check if a trained model exists for a persona."""
        model_dir = self._get_model_dir(persona_name)
        
        if not os.path.exists(model_dir):
            return {
                "status": "not_found",
                "persona_name": persona_name,
            }
        
        # Load metadata
        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        return {
            "status": "ready",
            "persona_name": persona_name,
            "model_dir": model_dir,
            "threshold": metadata["threshold"],
            "input_dim": metadata["input_dim"],
        }
    
    def get_persona_info(self, persona_name: str) -> dict:
        """Get information about a persona."""
        persona = get_persona(persona_name)
        return {
            "name": persona.name,
            "monthly_income": persona.monthly_income,
            "savings_goal": persona.savings_goal,
            "savings_rate": persona.savings_goal / persona.monthly_income,
            "monthly_budget": persona.monthly_income - persona.savings_goal,
        }


if __name__ == "__main__":
    # Test the client
    client = AccountantClient()
    
    print("Training model for balanced_ben...")
    result = client.train_persona_model("balanced_ben", days=60, epochs=20)
    print(f"Training complete: {result.to_dict()}")
    
    print("\nTesting scoring...")
    score = client.score_transaction(
        persona_name="balanced_ben",
        amount=150.0,
        category="dining",
        hour=19,
        day_of_week=5,
        days_to_month_end=10,
    )
    print(f"Score result: {score.to_dict()}")

