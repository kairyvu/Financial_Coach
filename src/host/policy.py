"""
Policy engine for Financial Coach.

Decides among three actions based on anomaly detection and context:
- do_nothing: No intervention needed
- gentle_nudge: Soft reminder about spending
- goal_reminder: Stronger reminder about savings goal
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional


class Action(Enum):
    """Possible actions the coach can take."""
    DO_NOTHING = "do_nothing"
    GENTLE_NUDGE = "gentle_nudge"
    GOAL_REMINDER = "goal_reminder"


@dataclass
class PolicyContext:
    """Context for policy decision making."""
    
    # Transaction details
    amount: float
    category: str
    is_anomaly: bool
    recon_error: float
    threshold: float
    
    # Goal tracking
    monthly_budget: float
    spent_this_month: float
    savings_goal: float
    
    # Temporal context
    days_to_month_end: int
    last_nudge_time: Optional[datetime] = None
    current_time: Optional[datetime] = None
    
    @property
    def goal_gap(self) -> float:
        """How much over/under budget we are (positive = over budget)."""
        expected_spent = self.monthly_budget * (1 - self.days_to_month_end / 30)
        return self.spent_this_month - expected_spent
    
    @property
    def goal_gap_ratio(self) -> float:
        """Goal gap as a ratio of monthly budget."""
        return self.goal_gap / self.monthly_budget if self.monthly_budget > 0 else 0
    
    @property
    def budget_remaining(self) -> float:
        """Remaining budget for the month."""
        return self.monthly_budget - self.spent_this_month
    
    @property
    def error_ratio(self) -> float:
        """Reconstruction error as ratio of threshold."""
        return self.recon_error / self.threshold if self.threshold > 0 else 0
    
    @property
    def is_end_of_month(self) -> bool:
        """True if within last 5 days of month."""
        return self.days_to_month_end <= 5
    
    @property
    def is_over_budget(self) -> bool:
        """True if spending is ahead of schedule."""
        return self.goal_gap > 0


@dataclass
class PolicyConfig:
    """Configuration for the policy engine."""
    
    # Cooldown between nudges (in hours)
    nudge_cooldown_hours: float = 4.0
    
    # Thresholds for escalation
    error_ratio_gentle: float = 1.0  # Above threshold = gentle nudge
    error_ratio_strong: float = 1.5  # 1.5x threshold = stronger nudge
    
    # Goal gap thresholds
    goal_gap_ratio_gentle: float = 0.05  # 5% over pace
    goal_gap_ratio_strong: float = 0.15  # 15% over pace
    
    # End of month sensitivity
    end_of_month_multiplier: float = 1.5  # More sensitive at month end
    
    # High-risk categories (more likely to nudge)
    high_risk_categories: tuple = ("entertainment", "shopping", "dining")
    
    # Amount thresholds (as ratio of daily budget)
    large_purchase_ratio: float = 0.5  # 50% of daily budget


class PolicyEngine:
    """
    Applies heuristic policy to decide on nudge actions.
    
    Policy logic:
    1. Check cooldown - skip if nudged recently
    2. If not anomalous and on track, do nothing
    3. If anomalous OR over budget, decide between gentle_nudge and goal_reminder
    4. Escalate to goal_reminder if:
       - Error is very high (1.5x threshold)
       - Significantly over budget (15%+)
       - End of month AND over budget
    """
    
    def __init__(self, config: Optional[PolicyConfig] = None):
        self.config = config or PolicyConfig()
    
    def _check_cooldown(self, context: PolicyContext) -> bool:
        """Check if we're in cooldown period. Returns True if should skip nudge."""
        if context.last_nudge_time is None:
            return False
        
        current = context.current_time or datetime.now()
        cooldown = timedelta(hours=self.config.nudge_cooldown_hours)
        
        return (current - context.last_nudge_time) < cooldown
    
    def _get_severity_score(self, context: PolicyContext) -> float:
        """
        Compute a severity score (0-1) for the transaction.
        
        Higher score = more likely to escalate to goal_reminder.
        """
        score = 0.0
        
        # Anomaly contribution (0-0.4)
        if context.is_anomaly:
            error_factor = min(context.error_ratio / 2.0, 1.0)  # Cap at 2x threshold
            score += 0.4 * error_factor
        
        # Goal gap contribution (0-0.3)
        if context.is_over_budget:
            gap_factor = min(context.goal_gap_ratio / 0.2, 1.0)  # Cap at 20% over
            score += 0.3 * gap_factor
        
        # End of month urgency (0-0.2)
        if context.is_end_of_month and context.is_over_budget:
            score += 0.2
        
        # High-risk category (0-0.1)
        if context.category in self.config.high_risk_categories:
            score += 0.1
        
        return min(score, 1.0)
    
    def decide(self, context: PolicyContext) -> tuple[Action, dict]:
        """
        Decide on an action based on context.
        
        Returns:
            Tuple of (Action, metadata dict)
        """
        metadata = {
            "is_anomaly": context.is_anomaly,
            "error_ratio": round(context.error_ratio, 3),
            "goal_gap_ratio": round(context.goal_gap_ratio, 3),
            "is_end_of_month": context.is_end_of_month,
            "is_over_budget": context.is_over_budget,
        }
        
        # Check cooldown first
        if self._check_cooldown(context):
            metadata["reason"] = "cooldown_active"
            return Action.DO_NOTHING, metadata
        
        # If not anomalous and on track, do nothing
        if not context.is_anomaly and not context.is_over_budget:
            metadata["reason"] = "on_track"
            return Action.DO_NOTHING, metadata
        
        # Compute severity score
        severity = self._get_severity_score(context)
        metadata["severity_score"] = round(severity, 3)
        
        # Decide based on severity
        if severity >= 0.6:
            metadata["reason"] = "high_severity"
            return Action.GOAL_REMINDER, metadata
        elif severity >= 0.3 or context.is_anomaly:
            metadata["reason"] = "moderate_severity"
            return Action.GENTLE_NUDGE, metadata
        else:
            metadata["reason"] = "low_severity"
            return Action.DO_NOTHING, metadata


# Singleton instance with default config
default_policy = PolicyEngine()

