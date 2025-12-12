"""Synthetic persona definitions for Financial Coach."""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Persona:
    """Represents a synthetic user persona with spending patterns."""
    
    name: str
    monthly_income: float
    savings_goal: float  # Monthly savings target
    category_weights: Dict[str, float]  # Spending distribution by category
    avg_daily_transactions: float
    spending_volatility: float  # 0-1, how variable spending is
    impulse_probability: float  # Probability of impulse purchases
    peak_hours: List[int]  # Hours when most active
    weekend_multiplier: float  # Spending multiplier on weekends


# Define 5 distinct personas for evaluation
PERSONAS = {
    "frugal_frank": Persona(
        name="frugal_frank",
        monthly_income=4500.0,
        savings_goal=900.0,  # 20% savings rate target
        category_weights={
            "groceries": 0.30,
            "utilities": 0.15,
            "transport": 0.12,
            "dining": 0.08,
            "entertainment": 0.05,
            "shopping": 0.10,
            "healthcare": 0.08,
            "subscriptions": 0.05,
            "misc": 0.07,
        },
        avg_daily_transactions=2.5,
        spending_volatility=0.2,
        impulse_probability=0.05,
        peak_hours=[12, 18, 19],
        weekend_multiplier=1.1,
    ),
    
    "spender_sarah": Persona(
        name="spender_sarah",
        monthly_income=6000.0,
        savings_goal=600.0,  # 10% savings rate target
        category_weights={
            "groceries": 0.15,
            "utilities": 0.08,
            "transport": 0.10,
            "dining": 0.20,
            "entertainment": 0.15,
            "shopping": 0.18,
            "healthcare": 0.04,
            "subscriptions": 0.05,
            "misc": 0.05,
        },
        avg_daily_transactions=4.0,
        spending_volatility=0.5,
        impulse_probability=0.25,
        peak_hours=[11, 14, 20, 21],
        weekend_multiplier=1.5,
    ),
    
    "balanced_ben": Persona(
        name="balanced_ben",
        monthly_income=5200.0,
        savings_goal=780.0,  # 15% savings rate target
        category_weights={
            "groceries": 0.22,
            "utilities": 0.12,
            "transport": 0.15,
            "dining": 0.12,
            "entertainment": 0.10,
            "shopping": 0.12,
            "healthcare": 0.06,
            "subscriptions": 0.06,
            "misc": 0.05,
        },
        avg_daily_transactions=3.0,
        spending_volatility=0.3,
        impulse_probability=0.12,
        peak_hours=[8, 12, 18],
        weekend_multiplier=1.2,
    ),
    
    "student_sam": Persona(
        name="student_sam",
        monthly_income=1800.0,
        savings_goal=180.0,  # 10% savings rate target
        category_weights={
            "groceries": 0.25,
            "utilities": 0.10,
            "transport": 0.08,
            "dining": 0.18,
            "entertainment": 0.15,
            "shopping": 0.08,
            "healthcare": 0.03,
            "subscriptions": 0.08,
            "misc": 0.05,
        },
        avg_daily_transactions=2.0,
        spending_volatility=0.4,
        impulse_probability=0.18,
        peak_hours=[10, 14, 22, 23],
        weekend_multiplier=1.4,
    ),
    
    "executive_emma": Persona(
        name="executive_emma",
        monthly_income=12000.0,
        savings_goal=3000.0,  # 25% savings rate target
        category_weights={
            "groceries": 0.10,
            "utilities": 0.05,
            "transport": 0.15,
            "dining": 0.20,
            "entertainment": 0.12,
            "shopping": 0.20,
            "healthcare": 0.08,
            "subscriptions": 0.05,
            "misc": 0.05,
        },
        avg_daily_transactions=3.5,
        spending_volatility=0.35,
        impulse_probability=0.15,
        peak_hours=[7, 12, 19, 20],
        weekend_multiplier=1.3,
    ),
}


def get_persona(name: str) -> Persona:
    """Get a persona by name."""
    if name not in PERSONAS:
        raise ValueError(f"Unknown persona: {name}. Available: {list(PERSONAS.keys())}")
    return PERSONAS[name]


def list_personas() -> List[str]:
    """List all available persona names."""
    return list(PERSONAS.keys())

