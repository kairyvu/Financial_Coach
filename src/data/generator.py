"""Synthetic transaction stream generator for Financial Coach."""

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional
import math

import numpy as np
import pandas as pd

from .personas import Persona, get_persona, list_personas


@dataclass
class Transaction:
    """Represents a single financial transaction."""
    
    id: str
    timestamp: datetime
    amount: float
    category: str
    persona_name: str
    is_anomaly: bool = False  # Ground truth for evaluation
    
    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame creation."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "amount": self.amount,
            "category": self.category,
            "persona_name": self.persona_name,
            "is_anomaly": self.is_anomaly,
            "hour": self.timestamp.hour,
            "day_of_week": self.timestamp.weekday(),
            "day_of_month": self.timestamp.day,
        }


@dataclass
class TransactionGenerator:
    """Generates synthetic transaction streams for a persona."""
    
    persona: Persona
    seed: Optional[int] = None
    _rng: np.random.Generator = field(init=False)
    _tx_counter: int = field(default=0, init=False)
    
    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)
    
    def _get_category_amount(self, category: str, base_daily_budget: float) -> float:
        """Generate amount for a category based on persona patterns."""
        weight = self.persona.category_weights.get(category, 0.05)
        base_amount = base_daily_budget * weight * 30 / self.persona.avg_daily_transactions
        
        # Add volatility
        volatility_factor = 1 + self._rng.normal(0, self.persona.spending_volatility)
        volatility_factor = max(0.3, min(2.5, volatility_factor))
        
        amount = base_amount * volatility_factor
        return round(max(1.0, amount), 2)
    
    def _get_transaction_hour(self, day_of_week: int) -> int:
        """Generate realistic transaction hour based on persona patterns."""
        # Weight peak hours more heavily
        hours = list(range(6, 24))  # 6 AM to 11 PM
        weights = []
        
        for h in hours:
            if h in self.persona.peak_hours:
                weights.append(3.0)
            elif abs(h - 12) <= 2 or abs(h - 19) <= 2:  # Lunch and dinner
                weights.append(2.0)
            else:
                weights.append(1.0)
        
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        return self._rng.choice(hours, p=weights)
    
    def _is_weekend(self, date: datetime) -> bool:
        """Check if date is weekend."""
        return date.weekday() >= 5
    
    def _generate_anomaly(self, normal_amount: float, category: str) -> tuple:
        """Generate an anomalous transaction (for ground truth labeling)."""
        anomaly_type = self._rng.choice(["high_amount", "unusual_category", "unusual_time"])
        
        if anomaly_type == "high_amount":
            # 2-5x normal amount
            multiplier = self._rng.uniform(2.5, 5.0)
            return round(normal_amount * multiplier, 2), category, True
        elif anomaly_type == "unusual_category":
            # Spend in a low-weight category
            low_weight_cats = [c for c, w in self.persona.category_weights.items() if w < 0.08]
            if low_weight_cats:
                new_cat = self._rng.choice(low_weight_cats)
                return round(normal_amount * 2, 2), new_cat, True
        
        return normal_amount, category, False
    
    def generate_stream(
        self,
        start_date: datetime,
        days: int = 60,
        anomaly_rate: float = 0.08,
    ) -> List[Transaction]:
        """
        Generate a stream of transactions for the specified period.
        
        Args:
            start_date: Start date for the stream
            days: Number of days to generate (60-90 recommended)
            anomaly_rate: Fraction of transactions that are anomalies
            
        Returns:
            List of Transaction objects
        """
        transactions = []
        monthly_budget = self.persona.monthly_income - self.persona.savings_goal
        daily_budget = monthly_budget / 30
        
        categories = list(self.persona.category_weights.keys())
        cat_weights = list(self.persona.category_weights.values())
        cat_weights = np.array(cat_weights) / sum(cat_weights)
        
        for day_offset in range(days):
            current_date = start_date + timedelta(days=day_offset)
            is_weekend = self._is_weekend(current_date)
            
            # Determine number of transactions for this day
            base_tx_count = self.persona.avg_daily_transactions
            if is_weekend:
                base_tx_count *= self.persona.weekend_multiplier
            
            # Add some daily variation
            tx_count = max(0, int(self._rng.poisson(base_tx_count)))
            
            for _ in range(tx_count):
                self._tx_counter += 1
                
                # Select category
                category = self._rng.choice(categories, p=cat_weights)
                
                # Generate amount
                amount = self._get_category_amount(category, daily_budget)
                if is_weekend:
                    amount *= self.persona.weekend_multiplier
                
                # Check for impulse purchase
                if self._rng.random() < self.persona.impulse_probability:
                    amount *= self._rng.uniform(1.5, 3.0)
                    amount = round(amount, 2)
                
                # Generate hour
                hour = self._get_transaction_hour(current_date.weekday())
                minute = self._rng.integers(0, 60)
                second = self._rng.integers(0, 60)
                
                timestamp = current_date.replace(hour=hour, minute=minute, second=second)
                
                # Determine if this should be an anomaly
                is_anomaly = False
                if self._rng.random() < anomaly_rate:
                    amount, category, is_anomaly = self._generate_anomaly(amount, category)
                
                tx = Transaction(
                    id=f"{self.persona.name}_{self._tx_counter:06d}",
                    timestamp=timestamp,
                    amount=amount,
                    category=category,
                    persona_name=self.persona.name,
                    is_anomaly=is_anomaly,
                )
                transactions.append(tx)
        
        # Sort by timestamp
        transactions.sort(key=lambda x: x.timestamp)
        return transactions


def generate_persona_dataset(
    persona_name: str,
    days: int = 60,
    seed: Optional[int] = None,
    start_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Generate a complete dataset for a persona.
    
    Args:
        persona_name: Name of the persona to generate data for
        days: Number of days of transactions
        seed: Random seed for reproducibility
        start_date: Start date (defaults to 60 days ago)
        
    Returns:
        DataFrame with transaction data
    """
    persona = get_persona(persona_name)
    generator = TransactionGenerator(persona=persona, seed=seed)
    
    if start_date is None:
        start_date = datetime.now() - timedelta(days=days)
    
    transactions = generator.generate_stream(start_date=start_date, days=days)
    
    df = pd.DataFrame([tx.to_dict() for tx in transactions])
    
    # Add derived features
    if not df.empty:
        df["days_to_month_end"] = df["timestamp"].apply(
            lambda x: (x.replace(day=28) + timedelta(days=4) - x).days
        )
        df["days_to_month_end"] = df["days_to_month_end"].clip(0, 31)
    
    return df


def generate_all_personas_dataset(
    days: int = 60,
    base_seed: int = 42,
) -> dict:
    """
    Generate datasets for all personas.
    
    Returns:
        Dictionary mapping persona names to DataFrames
    """
    datasets = {}
    for i, persona_name in enumerate(list_personas()):
        datasets[persona_name] = generate_persona_dataset(
            persona_name=persona_name,
            days=days,
            seed=base_seed + i,
        )
    return datasets


def save_datasets(datasets: dict, output_dir: str = "data/synthetic") -> None:
    """Save generated datasets to CSV files."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for persona_name, df in datasets.items():
        filepath = os.path.join(output_dir, f"{persona_name}_transactions.csv")
        df.to_csv(filepath, index=False)
        print(f"Saved {len(df)} transactions to {filepath}")


if __name__ == "__main__":
    # Generate and save datasets for all personas
    print("Generating synthetic transaction data...")
    datasets = generate_all_personas_dataset(days=75, base_seed=42)
    
    for name, df in datasets.items():
        anomaly_count = df["is_anomaly"].sum()
        print(f"{name}: {len(df)} transactions, {anomaly_count} anomalies ({100*anomaly_count/len(df):.1f}%)")
    
    save_datasets(datasets)
    print("\nDone!")

