#!/usr/bin/env python3
"""
Interactive Financial Coach - Input transactions and get nudges!
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.mcp_server.client import AccountantClient
from src.host.policy import PolicyEngine, PolicyContext, Action
from src.host.messages import MessageGenerator
from src.data.personas import get_persona, list_personas

# Available categories
CATEGORIES = [
    "groceries", "utilities", "transport", "dining",
    "entertainment", "shopping", "healthcare", "subscriptions", "misc"
]

def main():
    print("\n" + "="*60)
    print("  💰 Financial Coach - Interactive Mode")
    print("="*60)
    
    # Select persona
    print("\n📋 Available personas:")
    personas = list_personas()
    for i, p in enumerate(personas, 1):
        persona = get_persona(p)
        print(f"   {i}. {p} (Income: ${persona.monthly_income:,.0f}, "
              f"Savings Goal: ${persona.savings_goal:,.0f})")
    
    while True:
        try:
            choice = input("\nSelect persona (1-5) [default: 3 for balanced_ben]: ").strip()
            if choice == "":
                choice = "3"
            persona_idx = int(choice) - 1
            if 0 <= persona_idx < len(personas):
                persona_name = personas[persona_idx]
                break
            print("Invalid choice. Try again.")
        except ValueError:
            print("Please enter a number.")
    
    persona = get_persona(persona_name)
    monthly_budget = persona.monthly_income - persona.savings_goal
    
    print(f"\n✅ Selected: {persona_name}")
    print(f"   Monthly Budget: ${monthly_budget:,.0f}")
    print(f"   Savings Goal: ${persona.savings_goal:,.0f}")
    
    # Initialize components
    client = AccountantClient()
    
    # Check if model is trained
    status = client.get_model_status(persona_name)
    if status["status"] != "ready":
        print(f"\n⏳ Training model for {persona_name}...")
        client.train_persona_model(persona_name, verbose=False)
        print("✅ Model trained!")
    
    policy = PolicyEngine()
    messages = MessageGenerator()
    
    # Track spending
    spent_this_month = 0.0
    
    print("\n" + "-"*60)
    print("Enter transactions to get nudges. Type 'quit' to exit.")
    print("-"*60)
    
    while True:
        print("\n📝 Enter transaction details:")
        
        # Get amount
        amount_str = input("   Amount ($): ").strip()
        if amount_str.lower() == 'quit':
            break
        try:
            amount = float(amount_str.replace("$", "").replace(",", ""))
        except ValueError:
            print("   ❌ Invalid amount. Try again.")
            continue
        
        # Get category
        print(f"   Categories: {', '.join(CATEGORIES)}")
        category = input("   Category: ").strip().lower()
        if category not in CATEGORIES:
            print(f"   ❌ Invalid category. Using 'misc'.")
            category = "misc"
        
        # Get time info
        try:
            hour = input("   Hour (0-23) [default: 12]: ").strip()
            hour = int(hour) if hour else 12
            
            day = input("   Day of week (0=Mon, 6=Sun) [default: 3]: ").strip()
            day = int(day) if day else 3
            
            days_to_end = input("   Days to month end (0-31) [default: 15]: ").strip()
            days_to_end = int(days_to_end) if days_to_end else 15
        except ValueError:
            print("   Using default time values.")
            hour, day, days_to_end = 12, 3, 15
        
        # Score the transaction
        print("\n🔍 Analyzing transaction...")
        
        result = client.score_transaction(
            persona_name=persona_name,
            amount=amount,
            category=category,
            hour=hour,
            day_of_week=day,
            days_to_month_end=days_to_end,
        )
        
        # Update spending
        spent_this_month += amount
        
        # Build policy context
        context = PolicyContext(
            amount=amount,
            category=category,
            is_anomaly=result.is_anomaly,
            recon_error=result.recon_error,
            threshold=result.threshold,
            monthly_budget=monthly_budget,
            spent_this_month=spent_this_month,
            savings_goal=persona.savings_goal,
            days_to_month_end=days_to_end,
        )
        
        # Get action
        action, metadata = policy.decide(context)
        
        # Display results
        print("\n" + "="*60)
        
        if result.is_anomaly:
            print("⚠️  ANOMALY DETECTED!")
        else:
            print("✅ Normal transaction")
        
        print(f"   Error Ratio: {result.error_ratio:.2f}x threshold")
        print(f"   Budget Used: ${spent_this_month:,.0f} / ${monthly_budget:,.0f}")
        
        # Show nudge if any
        if action != Action.DO_NOTHING:
            message = messages.generate(action, context, amount, category)
            
            if action == Action.GENTLE_NUDGE:
                print(f"\n🔔 GENTLE NUDGE:")
            else:
                print(f"\n🚨 GOAL REMINDER:")
            
            print(f"   {message}")
        else:
            print("\n😊 No nudge needed - you're on track!")
        
        print("="*60)
    
    print("\n👋 Thanks for using Financial Coach!")
    print(f"   Total spent this session: ${spent_this_month:,.0f}")


if __name__ == "__main__":
    main()

