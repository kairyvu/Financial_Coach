"""
Message templates for Financial Coach nudges.

Generates supportive, short messages based on action type and context.
"""

import random
from typing import Optional

from .policy import Action, PolicyContext


# Template collections for each action type
GENTLE_NUDGE_TEMPLATES = [
    "Hey! That ${amount:.0f} on {category} is a bit unusual for you. Just a heads up! 💭",
    "Quick check-in: ${amount:.0f} on {category} today. Staying mindful? 🌟",
    "Noticed a ${amount:.0f} {category} purchase. Everything okay with the budget? 💪",
    "Just flagging: ${amount:.0f} on {category} is higher than your usual. No worries if planned! ✨",
    "Friendly reminder: That ${amount:.0f} {category} expense is on the larger side for you. 📊",
    "Heads up! ${amount:.0f} on {category} - want to keep an eye on that category? 👀",
]

GOAL_REMINDER_TEMPLATES = [
    "Important: You're ${gap:.0f} over your spending pace this month. That ${amount:.0f} on {category} might push your ${goal:.0f} savings goal. Consider waiting? 🎯",
    "Savings alert! With ${remaining:.0f} budget left and {days} days to go, this ${amount:.0f} {category} purchase could impact your ${goal:.0f} goal. 💰",
    "Hey, just want to help! You're ahead on spending by ${gap:.0f}. This ${amount:.0f} on {category} might make hitting your ${goal:.0f} savings target tougher. 🎯",
    "Quick math: ${remaining:.0f} left for {days} days. That's ${daily:.0f}/day. This ${amount:.0f} {category} purchase is significant. Still want to proceed? 📈",
    "Your ${goal:.0f} savings goal is at risk! You're ${gap:.0f} over pace with {days} days left. Maybe sleep on this ${amount:.0f} {category} purchase? 🌙",
]

END_OF_MONTH_TEMPLATES = [
    "Month-end stretch! Only {days} days left and you're ${gap:.0f} over budget. That ${amount:.0f} on {category} could hurt. Finish strong! 💪",
    "Final push! {days} days to go, ${remaining:.0f} remaining. This ${amount:.0f} {category} expense is risky for your ${goal:.0f} goal. 🏁",
    "Almost there! But ${gap:.0f} over with {days} days left. Consider skipping this ${amount:.0f} {category} purchase to protect your savings? 🎯",
]


class MessageGenerator:
    """Generates contextual nudge messages."""
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
    
    def generate(
        self,
        action: Action,
        context: PolicyContext,
        transaction_amount: float,
        category: str,
    ) -> Optional[str]:
        """
        Generate a message for the given action and context.
        
        Args:
            action: The action decided by the policy
            context: Policy context with goal/budget info
            transaction_amount: Amount of the transaction
            category: Category of the transaction
            
        Returns:
            Message string or None if no message needed
        """
        if action == Action.DO_NOTHING:
            return None
        
        # Prepare template variables
        vars = {
            "amount": transaction_amount,
            "category": category.replace("_", " "),
            "gap": abs(context.goal_gap),
            "remaining": max(0, context.budget_remaining),
            "days": context.days_to_month_end,
            "goal": context.savings_goal,
            "daily": context.budget_remaining / max(1, context.days_to_month_end),
        }
        
        # Select template based on action and context
        if action == Action.GOAL_REMINDER:
            if context.is_end_of_month and context.is_over_budget:
                templates = END_OF_MONTH_TEMPLATES
            else:
                templates = GOAL_REMINDER_TEMPLATES
        else:  # GENTLE_NUDGE
            templates = GENTLE_NUDGE_TEMPLATES
        
        template = self.rng.choice(templates)
        
        try:
            message = template.format(**vars)
        except KeyError:
            # Fallback if template has unexpected variables
            message = f"Heads up about that ${transaction_amount:.0f} {category} purchase! 💭"
        
        return message
    
    def generate_summary(
        self,
        total_spent: float,
        budget: float,
        savings_goal: float,
        days_remaining: int,
    ) -> str:
        """Generate a daily/weekly summary message."""
        remaining = budget - total_spent
        pace = total_spent / (30 - days_remaining) if days_remaining < 30 else 0
        target_pace = budget / 30
        
        if remaining < 0:
            return (
                f"⚠️ Budget exceeded by ${abs(remaining):.0f}! "
                f"Your ${savings_goal:.0f} savings goal is at risk. "
                f"Let's find ways to cut back these last {days_remaining} days."
            )
        elif pace > target_pace * 1.1:
            return (
                f"📊 Spending pace: ${pace:.0f}/day (target: ${target_pace:.0f}/day). "
                f"${remaining:.0f} left for {days_remaining} days. "
                f"Slight adjustment needed to hit your ${savings_goal:.0f} goal!"
            )
        else:
            return (
                f"✨ Great job! On track with ${remaining:.0f} remaining for {days_remaining} days. "
                f"Your ${savings_goal:.0f} savings goal is within reach!"
            )


# Default generator instance
default_generator = MessageGenerator(seed=42)


def generate_nudge_message(
    action: Action,
    context: PolicyContext,
    transaction_amount: float,
    category: str,
) -> Optional[str]:
    """Convenience function using default generator."""
    return default_generator.generate(action, context, transaction_amount, category)

