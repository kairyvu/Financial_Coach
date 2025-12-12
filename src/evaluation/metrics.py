"""
Evaluation metrics for Financial Coach.

Computes:
- Nudge precision/recall (anomaly detection accuracy)
- Savings uplift vs. rule-based baseline
- Nudges per week
- Latency metrics
"""

import time
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.host.controller import HostController, TransactionEvent
from src.host.policy import Action
from src.data.personas import get_persona


@dataclass
class EvaluationResult:
    """Complete evaluation results for a persona."""
    
    persona_name: str
    
    # Detection metrics
    precision: float
    recall: float
    f1_score: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    
    # Nudge metrics
    total_nudges: int
    gentle_nudges: int
    goal_reminders: int
    nudges_per_week: float
    
    # Savings metrics
    ae_savings_rate: float  # Estimated savings rate with AE nudges
    rule_savings_rate: float  # Savings rate with simple rule
    savings_uplift: float  # Percentage improvement
    
    # Latency
    avg_latency_ms: float
    p95_latency_ms: float
    
    def to_dict(self) -> dict:
        return {
            "persona_name": self.persona_name,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "true_negatives": self.true_negatives,
            "total_nudges": self.total_nudges,
            "gentle_nudges": self.gentle_nudges,
            "goal_reminders": self.goal_reminders,
            "nudges_per_week": round(self.nudges_per_week, 2),
            "ae_savings_rate": round(self.ae_savings_rate, 4),
            "rule_savings_rate": round(self.rule_savings_rate, 4),
            "savings_uplift": round(self.savings_uplift, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
        }


class RuleBasedBaseline:
    """
    Simple rule-based baseline for comparison.
    
    Rules:
    - Flag if amount > 2x average for category
    - Flag if amount > daily budget
    """
    
    def __init__(self, persona_name: str, training_df: pd.DataFrame):
        self.persona_name = persona_name
        self.persona = get_persona(persona_name)
        
        # Compute category averages from training data
        self.category_means = training_df.groupby("category")["amount"].mean().to_dict()
        self.global_mean = training_df["amount"].mean()
        
        # Daily budget
        monthly_budget = self.persona.monthly_income - self.persona.savings_goal
        self.daily_budget = monthly_budget / 30
    
    def is_anomaly(self, amount: float, category: str) -> bool:
        """Check if transaction is anomalous by simple rules."""
        cat_mean = self.category_means.get(category, self.global_mean)
        
        # Rule 1: Amount > 2x category average
        if amount > 2 * cat_mean:
            return True
        
        # Rule 2: Amount > daily budget
        if amount > self.daily_budget:
            return True
        
        return False
    
    def evaluate(self, df: pd.DataFrame) -> Tuple[float, float, float]:
        """
        Evaluate baseline on test data.
        
        Returns:
            Tuple of (precision, recall, f1)
        """
        predictions = df.apply(
            lambda row: self.is_anomaly(row["amount"], row["category"]),
            axis=1
        )
        
        ground_truth = df["is_anomaly"].values
        
        tp = ((predictions == True) & (ground_truth == True)).sum()
        fp = ((predictions == True) & (ground_truth == False)).sum()
        fn = ((predictions == False) & (ground_truth == True)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return precision, recall, f1


class Evaluator:
    """
    Evaluates the Financial Coach system against baselines.
    """
    
    def __init__(self):
        self.results: Dict[str, EvaluationResult] = {}
    
    def evaluate_persona(
        self,
        controller: HostController,
        test_df: pd.DataFrame,
        training_df: pd.DataFrame,
        verbose: bool = True,
    ) -> EvaluationResult:
        """
        Run full evaluation for a persona.
        
        Args:
            controller: Configured HostController
            test_df: Test transaction data
            training_df: Training data (for baseline)
            verbose: Print progress
            
        Returns:
            EvaluationResult with all metrics
        """
        persona_name = controller.persona_name
        
        if verbose:
            print(f"\n{'='*50}")
            print(f"Evaluating: {persona_name}")
            print(f"{'='*50}")
        
        # Track latencies
        latencies = []
        
        # Process all transactions
        decisions = []
        for idx, row in test_df.iterrows():
            event = TransactionEvent(
                id=row.get("id", f"tx_{idx}"),
                timestamp=pd.to_datetime(row["timestamp"]),
                amount=row["amount"],
                category=row["category"],
                persona_name=persona_name,
                is_anomaly_ground_truth=row.get("is_anomaly", False),
            )
            
            start_time = time.perf_counter()
            decision = controller.process_transaction(event)
            latency = (time.perf_counter() - start_time) * 1000
            latencies.append(latency)
            
            decisions.append({
                "ground_truth": event.is_anomaly_ground_truth,
                "predicted": decision.scoring_result.is_anomaly,
                "action": decision.action,
                "amount": event.amount,
            })
        
        decisions_df = pd.DataFrame(decisions)
        
        # Compute detection metrics
        tp = ((decisions_df["predicted"] == True) & (decisions_df["ground_truth"] == True)).sum()
        fp = ((decisions_df["predicted"] == True) & (decisions_df["ground_truth"] == False)).sum()
        fn = ((decisions_df["predicted"] == False) & (decisions_df["ground_truth"] == True)).sum()
        tn = ((decisions_df["predicted"] == False) & (decisions_df["ground_truth"] == False)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Compute nudge metrics
        action_counts = decisions_df["action"].value_counts()
        total_nudges = action_counts.get(Action.GENTLE_NUDGE, 0) + action_counts.get(Action.GOAL_REMINDER, 0)
        gentle_nudges = action_counts.get(Action.GENTLE_NUDGE, 0)
        goal_reminders = action_counts.get(Action.GOAL_REMINDER, 0)
        
        # Calculate weeks in test period
        timestamps = pd.to_datetime(test_df["timestamp"])
        days_span = (timestamps.max() - timestamps.min()).days + 1
        weeks = max(1, days_span / 7)
        nudges_per_week = total_nudges / weeks
        
        # Compute savings uplift estimation
        # Assumption: Each effective nudge prevents 30% of the flagged amount from being spent
        ae_prevented = decisions_df[decisions_df["predicted"] == True]["amount"].sum() * 0.3
        
        # Rule-based baseline
        baseline = RuleBasedBaseline(persona_name, training_df)
        rule_predictions = test_df.apply(
            lambda row: baseline.is_anomaly(row["amount"], row["category"]),
            axis=1
        )
        rule_prevented = test_df[rule_predictions == True]["amount"].sum() * 0.3
        
        # Calculate savings rates
        total_spent = test_df["amount"].sum()
        persona = get_persona(persona_name)
        monthly_budget = persona.monthly_income - persona.savings_goal
        
        # Normalize to monthly
        months = max(1, days_span / 30)
        monthly_spent = total_spent / months
        
        # Calculate effective savings with nudge-prevented spending
        ae_effective_savings = persona.monthly_income - monthly_spent + ae_prevented/months
        rule_effective_savings = persona.monthly_income - monthly_spent + rule_prevented/months
        
        ae_savings_rate = ae_effective_savings / persona.monthly_income
        rule_savings_rate = rule_effective_savings / persona.monthly_income
        
        # Uplift is the relative improvement in savings
        # If both are negative (over budget), compare the reduction in overspending
        if ae_prevented > rule_prevented:
            savings_uplift = (ae_prevented - rule_prevented) / max(rule_prevented, 1)
        else:
            savings_uplift = (ae_prevented - rule_prevented) / max(rule_prevented, 1) if rule_prevented > 0 else 0
        
        # Latency metrics
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        
        result = EvaluationResult(
            persona_name=persona_name,
            precision=precision,
            recall=recall,
            f1_score=f1,
            true_positives=int(tp),
            false_positives=int(fp),
            false_negatives=int(fn),
            true_negatives=int(tn),
            total_nudges=int(total_nudges),
            gentle_nudges=int(gentle_nudges),
            goal_reminders=int(goal_reminders),
            nudges_per_week=nudges_per_week,
            ae_savings_rate=ae_savings_rate,
            rule_savings_rate=rule_savings_rate,
            savings_uplift=savings_uplift,
            avg_latency_ms=avg_latency,
            p95_latency_ms=p95_latency,
        )
        
        self.results[persona_name] = result
        
        if verbose:
            self._print_result(result)
        
        return result
    
    def _print_result(self, result: EvaluationResult):
        """Print formatted results."""
        print(f"\n📊 Detection Metrics:")
        print(f"   Precision: {result.precision:.3f}")
        print(f"   Recall:    {result.recall:.3f}")
        print(f"   F1 Score:  {result.f1_score:.3f}")
        print(f"   (TP={result.true_positives}, FP={result.false_positives}, "
              f"FN={result.false_negatives}, TN={result.true_negatives})")
        
        print(f"\n💬 Nudge Metrics:")
        print(f"   Total Nudges:    {result.total_nudges}")
        print(f"   Gentle Nudges:   {result.gentle_nudges}")
        print(f"   Goal Reminders:  {result.goal_reminders}")
        print(f"   Nudges/Week:     {result.nudges_per_week:.1f}")
        
        print(f"\n💰 Savings Metrics:")
        print(f"   AE Savings Rate:   {result.ae_savings_rate:.1%}")
        print(f"   Rule Savings Rate: {result.rule_savings_rate:.1%}")
        print(f"   Savings Uplift:    {result.savings_uplift:+.1%}")
        
        print(f"\n⏱️ Latency:")
        print(f"   Average: {result.avg_latency_ms:.2f}ms")
        print(f"   P95:     {result.p95_latency_ms:.2f}ms")
    
    def get_summary_df(self) -> pd.DataFrame:
        """Get summary of all results as DataFrame."""
        if not self.results:
            return pd.DataFrame()
        
        records = [r.to_dict() for r in self.results.values()]
        return pd.DataFrame(records)
    
    def save_results(self, output_dir: str = "results"):
        """Save all results to files."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save CSV
        df = self.get_summary_df()
        csv_path = os.path.join(output_dir, "evaluation_metrics.csv")
        df.to_csv(csv_path, index=False)
        
        # Save markdown report
        md_path = os.path.join(output_dir, "evaluation_report.md")
        self._save_markdown_report(md_path)
        
        print(f"\n✅ Results saved to {output_dir}/")
    
    def _save_markdown_report(self, filepath: str):
        """Generate markdown report."""
        df = self.get_summary_df()
        
        report = [
            "# Financial Coach Evaluation Report",
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            "",
        ]
        
        # Overall averages
        avg_precision = df["precision"].mean()
        avg_recall = df["recall"].mean()
        avg_f1 = df["f1_score"].mean()
        avg_uplift = df["savings_uplift"].mean()
        
        report.extend([
            f"- **Average Precision**: {avg_precision:.3f}",
            f"- **Average Recall**: {avg_recall:.3f}",
            f"- **Average F1 Score**: {avg_f1:.3f}",
            f"- **Average Savings Uplift vs Rule Baseline**: {avg_uplift:+.1%}",
            "",
            "## Per-Persona Results",
            "",
        ])
        
        # Table header
        report.append("| Persona | Precision | Recall | F1 | Nudges/Week | Savings Uplift |")
        report.append("|---------|-----------|--------|-----|-------------|----------------|")
        
        for _, row in df.iterrows():
            report.append(
                f"| {row['persona_name']} | {row['precision']:.3f} | {row['recall']:.3f} | "
                f"{row['f1_score']:.3f} | {row['nudges_per_week']:.1f} | {row['savings_uplift']:+.1%} |"
            )
        
        report.extend([
            "",
            "## Latency",
            "",
            f"- **Average Latency**: {df['avg_latency_ms'].mean():.2f}ms",
            f"- **P95 Latency**: {df['p95_latency_ms'].mean():.2f}ms",
            "",
            "## Methodology",
            "",
            "- Anomaly detection uses a dense autoencoder trained per-persona",
            "- Threshold set at 95th percentile of validation reconstruction errors",
            "- Rule baseline flags transactions > 2x category average or > daily budget",
            "- Savings uplift assumes 30% of flagged transaction amount is prevented",
        ])
        
        with open(filepath, "w") as f:
            f.write("\n".join(report))

