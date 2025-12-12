"""Autoencoder model for transaction anomaly detection."""

import json
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Transaction categories
CATEGORIES = [
    "groceries", "utilities", "transport", "dining", 
    "entertainment", "shopping", "healthcare", "subscriptions", "misc"
]


class TransactionAutoencoder(nn.Module):
    """
    Dense Autoencoder for transaction anomaly detection.
    
    Architecture: input_dim -> 32 -> 8 (bottleneck) -> 32 -> input_dim
    """
    
    def __init__(self, input_dim: int):
        super().__init__()
        
        self.input_dim = input_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder and decoder."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Compute MSE reconstruction error for each sample."""
        with torch.no_grad():
            reconstructed = self.forward(x)
            mse = torch.mean((x - reconstructed) ** 2, dim=1)
        return mse


@dataclass
class FeatureProcessor:
    """Processes raw transaction features for the autoencoder."""
    
    amount_scaler: Optional[StandardScaler] = None
    category_encoder: Optional[OneHotEncoder] = None
    fitted: bool = False
    
    def fit(self, df: pd.DataFrame) -> "FeatureProcessor":
        """Fit the preprocessors on training data."""
        # Amount z-scoring
        self.amount_scaler = StandardScaler()
        self.amount_scaler.fit(df[["amount"]])
        
        # Category one-hot encoding
        self.category_encoder = OneHotEncoder(
            categories=[CATEGORIES],
            sparse_output=False,
            handle_unknown="ignore"
        )
        self.category_encoder.fit(df[["category"]])
        
        self.fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform transaction data to feature vectors.
        
        Features:
        - amount (z-scored)
        - category (one-hot, 9 dims)
        - hour (sin/cos, 2 dims)
        - day_of_week (sin/cos, 2 dims)
        - days_to_month_end (scaled 0-1)
        
        Total: 1 + 9 + 2 + 2 + 1 = 15 dimensions
        """
        if not self.fitted:
            raise ValueError("FeatureProcessor must be fitted before transform")
        
        features = []
        
        # Amount (z-scored)
        amount_scaled = self.amount_scaler.transform(df[["amount"]])
        features.append(amount_scaled)
        
        # Category (one-hot)
        category_encoded = self.category_encoder.transform(df[["category"]])
        features.append(category_encoded)
        
        # Hour (sin/cos encoding for cyclical feature)
        hour = df["hour"].values
        hour_sin = np.sin(2 * np.pi * hour / 24).reshape(-1, 1)
        hour_cos = np.cos(2 * np.pi * hour / 24).reshape(-1, 1)
        features.append(hour_sin)
        features.append(hour_cos)
        
        # Day of week (sin/cos encoding)
        dow = df["day_of_week"].values
        dow_sin = np.sin(2 * np.pi * dow / 7).reshape(-1, 1)
        dow_cos = np.cos(2 * np.pi * dow / 7).reshape(-1, 1)
        features.append(dow_sin)
        features.append(dow_cos)
        
        # Days to month end (scaled 0-1)
        days_to_end = df["days_to_month_end"].values.reshape(-1, 1) / 31.0
        features.append(days_to_end)
        
        return np.hstack(features).astype(np.float32)
    
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(df)
        return self.transform(df)
    
    def save(self, filepath: str) -> None:
        """Save processor state to file."""
        import pickle
        state = {
            "amount_scaler": self.amount_scaler,
            "category_encoder": self.category_encoder,
            "fitted": self.fitted,
        }
        with open(filepath, "wb") as f:
            pickle.dump(state, f)
    
    @classmethod
    def load(cls, filepath: str) -> "FeatureProcessor":
        """Load processor from file."""
        import pickle
        with open(filepath, "rb") as f:
            state = pickle.load(f)
        
        processor = cls()
        processor.amount_scaler = state["amount_scaler"]
        processor.category_encoder = state["category_encoder"]
        processor.fitted = state["fitted"]
        return processor


class AnomalyDetector:
    """
    Complete anomaly detection pipeline for a persona.
    
    Combines feature processing, autoencoder model, and threshold-based detection.
    """
    
    def __init__(
        self,
        persona_name: str,
        threshold_percentile: float = 95.0,
    ):
        self.persona_name = persona_name
        self.threshold_percentile = threshold_percentile
        
        self.processor: Optional[FeatureProcessor] = None
        self.model: Optional[TransactionAutoencoder] = None
        self.threshold: Optional[float] = None
        self.input_dim: Optional[int] = None
        self.trained: bool = False
    
    def train(
        self,
        df: pd.DataFrame,
        epochs: int = 30,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        validation_split: float = 0.2,
        early_stopping_patience: int = 5,
        verbose: bool = True,
    ) -> dict:
        """
        Train the autoencoder on transaction data.
        
        Args:
            df: Transaction DataFrame
            epochs: Maximum training epochs
            batch_size: Training batch size
            learning_rate: Adam learning rate
            validation_split: Fraction of data for validation
            early_stopping_patience: Epochs without improvement before stopping
            verbose: Print training progress
            
        Returns:
            Training metrics dictionary
        """
        # Fit feature processor and transform data
        self.processor = FeatureProcessor()
        X = self.processor.fit_transform(df)
        self.input_dim = X.shape[1]
        
        # Split into train/validation
        n_samples = len(X)
        n_val = int(n_samples * validation_split)
        indices = np.random.permutation(n_samples)
        
        train_indices = indices[n_val:]
        val_indices = indices[:n_val]
        
        X_train = torch.FloatTensor(X[train_indices])
        X_val = torch.FloatTensor(X[val_indices])
        
        # Initialize model
        self.model = TransactionAutoencoder(self.input_dim)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop with early stopping
        best_val_loss = float("inf")
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            # Mini-batch training
            perm = torch.randperm(len(X_train))
            for i in range(0, len(X_train), batch_size):
                batch_indices = perm[i:i + batch_size]
                batch = X_train[batch_indices]
                
                optimizer.zero_grad()
                output = self.model(batch)
                loss = criterion(output, batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * len(batch)
            
            train_loss /= len(X_train)
            train_losses.append(train_loss)
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_output = self.model(X_val)
                val_loss = criterion(val_output, X_val).item()
            val_losses.append(val_loss)
            
            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break
        
        # Compute threshold on validation set (95th percentile)
        self.model.eval()
        val_errors = self.model.get_reconstruction_error(X_val).numpy()
        self.threshold = float(np.percentile(val_errors, self.threshold_percentile))
        
        self.trained = True
        
        if verbose:
            print(f"Training complete. Threshold (p{self.threshold_percentile}): {self.threshold:.6f}")
        
        return {
            "epochs_trained": epoch + 1,
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
            "threshold": self.threshold,
            "train_losses": train_losses,
            "val_losses": val_losses,
        }
    
    def score(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Score transactions for anomalies.
        
        Args:
            df: Transaction DataFrame
            
        Returns:
            Tuple of (reconstruction_errors, is_anomaly_flags)
        """
        if not self.trained:
            raise ValueError("Model must be trained before scoring")
        
        X = self.processor.transform(df)
        X_tensor = torch.FloatTensor(X)
        
        self.model.eval()
        errors = self.model.get_reconstruction_error(X_tensor).numpy()
        is_anomaly = errors > self.threshold
        
        return errors, is_anomaly
    
    def score_single(self, transaction: dict) -> dict:
        """
        Score a single transaction.
        
        Args:
            transaction: Dictionary with transaction fields
            
        Returns:
            Dictionary with recon_error and is_anomaly
        """
        df = pd.DataFrame([transaction])
        errors, is_anomaly = self.score(df)
        
        return {
            "recon_error": float(errors[0]),
            "is_anomaly": bool(is_anomaly[0]),
            "threshold": self.threshold,
        }
    
    def save(self, output_dir: str) -> None:
        """Save all model artifacts to directory."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model weights
        model_path = os.path.join(output_dir, "model.pt")
        torch.save(self.model.state_dict(), model_path)
        
        # Save feature processor
        processor_path = os.path.join(output_dir, "processor.pkl")
        self.processor.save(processor_path)
        
        # Save metadata
        metadata = {
            "persona_name": self.persona_name,
            "threshold": self.threshold,
            "threshold_percentile": self.threshold_percentile,
            "input_dim": self.input_dim,
        }
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved model artifacts to {output_dir}")
    
    @classmethod
    def load(cls, model_dir: str) -> "AnomalyDetector":
        """Load a trained model from directory."""
        # Load metadata
        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        detector = cls(
            persona_name=metadata["persona_name"],
            threshold_percentile=metadata["threshold_percentile"],
        )
        detector.threshold = metadata["threshold"]
        detector.input_dim = metadata["input_dim"]
        
        # Load feature processor
        processor_path = os.path.join(model_dir, "processor.pkl")
        detector.processor = FeatureProcessor.load(processor_path)
        
        # Load model
        detector.model = TransactionAutoencoder(detector.input_dim)
        model_path = os.path.join(model_dir, "model.pt")
        detector.model.load_state_dict(torch.load(model_path, weights_only=True))
        detector.model.eval()
        
        detector.trained = True
        return detector


if __name__ == "__main__":
    # Test the autoencoder on synthetic data
    from src.data.generator import generate_persona_dataset
    
    print("Testing Autoencoder on synthetic data...")
    
    # Generate test data
    df = generate_persona_dataset("balanced_ben", days=60, seed=42)
    print(f"Generated {len(df)} transactions")
    
    # Train detector
    detector = AnomalyDetector(persona_name="balanced_ben")
    metrics = detector.train(df, epochs=30, verbose=True)
    
    # Test scoring
    errors, is_anomaly = detector.score(df)
    print(f"\nDetected {is_anomaly.sum()} anomalies out of {len(df)} transactions")
    
    # Test single transaction scoring
    sample_tx = df.iloc[0].to_dict()
    result = detector.score_single(sample_tx)
    print(f"\nSingle transaction score: {result}")
    
    # Save and reload test
    detector.save("artifacts/test_model")
    loaded = AnomalyDetector.load("artifacts/test_model")
    result2 = loaded.score_single(sample_tx)
    print(f"Loaded model score: {result2}")

