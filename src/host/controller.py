"""
Host Controller for Financial Coach.

Orchestrates the sense-and-respond loop:
1. Ingest transactions
2. Call Accountant (MCP server) for anomaly detection
3. Apply policy to decide on action
4. Generate and emit messages
5. Log outcomes for evaluation
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Generator

import pandas as pd

from src.mcp_server.client import AccountantClient, ScoringResult
from src.data.personas import get_persona, Persona
from .policy import PolicyEngine, PolicyContext, PolicyConfig, Action
from .messages import MessageGenerator, generate_nudge_message


@dataclass
class TransactionEvent:
    """A transaction event to process."""
    id: str
    timestamp: datetime
    amount: float
    category: str
    persona_name: str
    is_anomaly_ground_truth: bool = False  # For evaluation


@dataclass
class NudgeDecision:
    """Record of a nudge decision."""
    transaction_id: str
    timestamp: datetime
    action: Action
    message: Optional[str]
    metadata: dict
    scoring_result: ScoringResult
    
    def to_dict(self) -> dict:
        return {
            "transaction_id": self.transaction_id,
            "timestamp": self.timestamp.isoformat(),
            "action": self.action.value,
            "message": self.message,
            "is_anomaly_predicted": self.scoring_result.is_anomaly,
            "recon_error": self.scoring_result.recon_error,
            "threshold": self.scoring_result.threshold,
            "error_ratio": self.scoring_result.error_ratio,
            **self.metadata,
        }


@dataclass
class HostState:
    """Tracks the state of the host controller."""
    persona_name: str
    monthly_budget: float
    savings_goal: float
    
    # Tracking
    spent_this_month: float = 0.0
    current_month: Optional[int] = None
    last_nudge_time: Optional[datetime] = None
    
    # History
    decisions: List[NudgeDecision] = field(default_factory=list)
    transactions_processed: int = 0
    nudges_sent: int = 0
    
    def reset_month(self, month: int):
        """Reset monthly tracking."""
        self.current_month = month
        self.spent_this_month = 0.0
    
    def add_spending(self, amount: float, timestamp: datetime):
        """Add spending and handle month transitions."""
        if self.current_month is None or timestamp.month != self.current_month:
            self.reset_month(timestamp.month)
        self.spent_this_month += amount
    
    def record_nudge(self, timestamp: datetime):
        """Record that a nudge was sent."""
        self.last_nudge_time = timestamp
        self.nudges_sent += 1


class HostController:
    """
    Main controller for the Financial Coach system.
    
    Coordinates between:
    - Transaction ingestion
    - Accountant (anomaly detection)
    - Policy engine (action decision)
    - Message generation
    - Outcome logging
    """
    
    def __init__(
        self,
        persona_name: str,
        accountant: Optional[AccountantClient] = None,
        policy: Optional[PolicyEngine] = None,
        message_generator: Optional[MessageGenerator] = None,
    ):
        self.persona_name = persona_name
        self.persona = get_persona(persona_name)
        
        # Components
        self.accountant = accountant or AccountantClient()
        self.policy = policy or PolicyEngine()
        self.message_generator = message_generator or MessageGenerator()
        
        # Initialize state
        monthly_budget = self.persona.monthly_income - self.persona.savings_goal
        self.state = HostState(
            persona_name=persona_name,
            monthly_budget=monthly_budget,
            savings_goal=self.persona.savings_goal,
        )
    
    def ensure_model_trained(self, days: int = 60, epochs: int = 30) -> bool:
        """Ensure the persona model is trained."""
        status = self.accountant.get_model_status(self.persona_name)
        
        if status["status"] == "ready":
            return True
        
        print(f"Training model for {self.persona_name}...")
        result = self.accountant.train_persona_model(
            self.persona_name,
            days=days,
            epochs=epochs,
        )
        print(f"Training complete. Threshold: {result.threshold:.6f}")
        return True
    
    def process_transaction(
        self,
        event: TransactionEvent,
    ) -> NudgeDecision:
        """
        Process a single transaction through the sense-and-respond loop.
        
        Args:
            event: Transaction event to process
            
        Returns:
            NudgeDecision with action taken
        """
        # Update spending tracking
        self.state.add_spending(event.amount, event.timestamp)
        self.state.transactions_processed += 1
        
        # Calculate days to month end
        days_in_month = 30  # Simplified
        days_to_month_end = max(0, days_in_month - event.timestamp.day)
        
        # Step 1: Call Accountant for anomaly detection
        scoring_result = self.accountant.score_transaction(
            persona_name=self.persona_name,
            amount=event.amount,
            category=event.category,
            hour=event.timestamp.hour,
            day_of_week=event.timestamp.weekday(),
            days_to_month_end=days_to_month_end,
        )
        
        # Step 2: Build policy context
        context = PolicyContext(
            amount=event.amount,
            category=event.category,
            is_anomaly=scoring_result.is_anomaly,
            recon_error=scoring_result.recon_error,
            threshold=scoring_result.threshold,
            monthly_budget=self.state.monthly_budget,
            spent_this_month=self.state.spent_this_month,
            savings_goal=self.state.savings_goal,
            days_to_month_end=days_to_month_end,
            last_nudge_time=self.state.last_nudge_time,
            current_time=event.timestamp,
        )
        
        # Step 3: Apply policy
        action, metadata = self.policy.decide(context)
        
        # Step 4: Generate message
        message = None
        if action != Action.DO_NOTHING:
            message = self.message_generator.generate(
                action=action,
                context=context,
                transaction_amount=event.amount,
                category=event.category,
            )
            self.state.record_nudge(event.timestamp)
        
        # Step 5: Create and record decision
        decision = NudgeDecision(
            transaction_id=event.id,
            timestamp=event.timestamp,
            action=action,
            message=message,
            metadata=metadata,
            scoring_result=scoring_result,
        )
        
        self.state.decisions.append(decision)
        
        return decision
    
    def replay_transactions(
        self,
        df: pd.DataFrame,
        verbose: bool = True,
    ) -> Generator[NudgeDecision, None, None]:
        """
        Replay a stream of transactions.
        
        Args:
            df: DataFrame with transaction data
            verbose: Print progress and nudges
            
        Yields:
            NudgeDecision for each transaction
        """
        for idx, row in df.iterrows():
            event = TransactionEvent(
                id=row.get("id", f"tx_{idx}"),
                timestamp=pd.to_datetime(row["timestamp"]),
                amount=row["amount"],
                category=row["category"],
                persona_name=self.persona_name,
                is_anomaly_ground_truth=row.get("is_anomaly", False),
            )
            
            decision = self.process_transaction(event)
            
            if verbose and decision.message:
                print(f"\n[{event.timestamp.strftime('%Y-%m-%d %H:%M')}] "
                      f"${event.amount:.2f} {event.category}")
                print(f"  → {decision.message}")
            
            yield decision
    
    def get_decisions_df(self) -> pd.DataFrame:
        """Get all decisions as a DataFrame."""
        if not self.state.decisions:
            return pd.DataFrame()
        
        records = [d.to_dict() for d in self.state.decisions]
        return pd.DataFrame(records)
    
    def get_summary(self) -> dict:
        """Get a summary of the session."""
        decisions_df = self.get_decisions_df()
        
        if decisions_df.empty:
            return {"transactions": 0, "nudges": 0}
        
        action_counts = decisions_df["action"].value_counts().to_dict()
        
        return {
            "persona": self.persona_name,
            "transactions_processed": self.state.transactions_processed,
            "nudges_sent": self.state.nudges_sent,
            "action_counts": action_counts,
            "anomalies_detected": int(decisions_df["is_anomaly_predicted"].sum()),
            "spent_current_month": round(self.state.spent_this_month, 2),
            "budget": self.state.monthly_budget,
            "savings_goal": self.state.savings_goal,
        }
    
    def save_log(self, filepath: str) -> None:
        """Save decision log to file."""
        df = self.get_decisions_df()
        
        if filepath.endswith(".csv"):
            df.to_csv(filepath, index=False)
        else:
            df.to_json(filepath, orient="records", indent=2, date_format="iso")
        
        print(f"Saved {len(df)} decisions to {filepath}")


def create_controller(
    persona_name: str,
    train_if_needed: bool = True,
    **kwargs,
) -> HostController:
    """
    Factory function to create a configured controller.
    
    Args:
        persona_name: Name of the persona
        train_if_needed: Auto-train model if not found
        **kwargs: Additional arguments for HostController
        
    Returns:
        Configured HostController
    """
    controller = HostController(persona_name, **kwargs)
    
    if train_if_needed:
        controller.ensure_model_trained()
    
    return controller

