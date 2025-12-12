# Financial Coach 🎯💰

An AI-powered savings coach that helps users reach their financial goals through timely, contextual nudges. The system uses a tiny Autoencoder for anomaly detection to identify unusual spending patterns and provides supportive messages to keep users on track.

## Overview

Financial Coach implements an end-to-end **sense-and-respond loop**:

1. **Sense**: Detect unusual transactions using an Autoencoder trained on user spending patterns
2. **Decide**: Apply a heuristic policy to choose an appropriate action
3. **Respond**: Generate supportive, contextual nudge messages

The prototype demonstrates that AE-driven nudges outperform simple rule-based thresholds for detecting spending anomalies.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Host Controller (CLI)                     │
│  • Ingests transactions                                      │
│  • Applies policy (do_nothing / gentle_nudge / goal_reminder)│
│  • Generates templated messages                              │
│  • Logs decisions for evaluation                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Accountant (MCP Server)                     │
│  • Trains persona-specific autoencoders                      │
│  • Scores transactions for anomalies                         │
│  • Returns: recon_error, is_anomaly, threshold               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Autoencoder Model                         │
│  • Dense AE: input → 32 → 8 (bottleneck) → 32 → input       │
│  • Features: amount, category, hour, day_of_week, days_to_end│
│  • Threshold: 95th percentile of reconstruction errors       │
└─────────────────────────────────────────────────────────────┘
```

## Features

- **5 Synthetic Personas**: Diverse spending profiles for evaluation
  - `frugal_frank` - Conservative spender, 20% savings rate
  - `spender_sarah` - High spender, impulse buyer
  - `balanced_ben` - Moderate, balanced spending
  - `student_sam` - Limited budget, occasional splurges
  - `executive_emma` - High income, high savings target

- **Autoencoder Anomaly Detection**: Learns normal spending patterns per user
- **Contextual Policy Engine**: Considers goal progress, time of month, cooldowns
- **Supportive Messaging**: Non-judgmental, encouraging nudge templates
- **MCP Server**: Exposes training and scoring as MCP tools
- **Comprehensive Evaluation**: Precision/recall, savings uplift, latency metrics

## Quick Start

### Installation

```bash
# Clone the repository
cd Financial_Coach

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Demo

```bash
# Basic demo with default persona (balanced_ben)
python demo.py

# Demo with specific persona
python demo.py --persona spender_sarah

# Demo all personas
python demo.py --all

# Process more days of data
python demo.py --days 60
```

### Sample Output

```
============================================================
 Financial Coach Demo - balanced_ben
============================================================

👤 Persona: balanced_ben
   Monthly Income:  $5,200
   Savings Goal:    $780 (15%)
   Monthly Budget:  $4,420
   Spending Style:  Moderate

------------------------------------------------------------
 Step 3: Processing Transactions (Sense & Respond)
------------------------------------------------------------

🔄 Processing transaction stream...
   (Showing nudges only)

   🔔 [2024-11-15 19:23]
      Hey! That $245 on shopping is a bit unusual for you. Just a heads up! 💭

   🚨 [2024-11-18 12:45]
      Savings alert! With $2,150 budget left and 12 days to go, this $380 
      entertainment purchase could impact your $780 goal. 💰

------------------------------------------------------------
 Step 4: Session Summary
------------------------------------------------------------

📊 Results:
   Transactions Processed: 89
   Anomalies Detected:     12
   Total Nudges Sent:      8

🎯 Detection Accuracy:
   Precision: 78.5%
   Recall:    65.2%
```

## Run Full Evaluation

```bash
# Run evaluation across all personas
python -m src.evaluation.run_evaluation

# Customize evaluation parameters
python -m src.evaluation.run_evaluation --train-days 75 --test-days 30 --epochs 40

# Output saved to results/ directory
```

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Precision | Fraction of flagged transactions that are true anomalies |
| Recall | Fraction of true anomalies that were detected |
| F1 Score | Harmonic mean of precision and recall |
| Nudges/Week | Average number of nudges sent per week |
| Savings Uplift | Improvement vs. rule-based baseline |
| Latency | Processing time per transaction |

## Project Structure

```
Financial_Coach/
├── demo.py                    # Demo script
├── requirements.txt           # Dependencies
├── README.md
│
├── src/
│   ├── data/
│   │   ├── personas.py        # Synthetic persona definitions
│   │   └── generator.py       # Transaction stream generator
│   │
│   ├── models/
│   │   └── autoencoder.py     # AE model and feature processing
│   │
│   ├── mcp_server/
│   │   ├── accountant.py      # MCP server implementation
│   │   └── client.py          # Direct Python client
│   │
│   ├── host/
│   │   ├── controller.py      # Host controller (orchestration)
│   │   ├── policy.py          # Policy engine (action decisions)
│   │   └── messages.py        # Message templates
│   │
│   └── evaluation/
│       ├── metrics.py         # Evaluation metrics
│       └── run_evaluation.py  # Full evaluation script
│
├── artifacts/                 # Trained model artifacts (generated)
│   └── {persona_name}/
│       ├── model.pt
│       ├── processor.pkl
│       └── metadata.json
│
└── results/                   # Evaluation results (generated)
    ├── evaluation_metrics.csv
    └── evaluation_report.md
```

## MCP Server Usage

The Accountant MCP server exposes two main tools:

### 1. Train Persona Model

```python
from src.mcp_server.client import AccountantClient

client = AccountantClient()
result = client.train_persona_model(
    persona_name="balanced_ben",
    days=60,
    epochs=30,
)
print(f"Threshold: {result.threshold}")
```

### 2. Score Transaction

```python
score = client.score_transaction(
    persona_name="balanced_ben",
    amount=150.0,
    category="dining",
    hour=19,
    day_of_week=5,
    days_to_month_end=10,
)
print(f"Anomaly: {score.is_anomaly}, Error: {score.recon_error:.4f}")
```

### Running as MCP Server

```bash
# Start the MCP server (stdio transport)
python -m src.mcp_server.accountant
```

## Policy Actions

The policy engine decides among three actions:

| Action | Trigger | Example Message |
|--------|---------|-----------------|
| `do_nothing` | Normal transaction, on track | (no message) |
| `gentle_nudge` | Anomalous but not critical | "Hey! That $150 on dining is a bit unusual for you. Just a heads up! 💭" |
| `goal_reminder` | High severity, goal at risk | "Savings alert! With $800 left and 5 days to go, this $200 shopping purchase could impact your $500 goal. 💰" |

## Configuration

### Policy Configuration

```python
from src.host.policy import PolicyConfig, PolicyEngine

config = PolicyConfig(
    nudge_cooldown_hours=4.0,      # Hours between nudges
    error_ratio_gentle=1.0,         # Threshold for gentle nudge
    error_ratio_strong=1.5,         # Threshold for goal reminder
    goal_gap_ratio_gentle=0.05,     # 5% over pace triggers nudge
    goal_gap_ratio_strong=0.15,     # 15% over pace escalates
)

policy = PolicyEngine(config)
```

### Custom Personas

```python
from src.data.personas import Persona

my_persona = Persona(
    name="custom_user",
    monthly_income=5000.0,
    savings_goal=1000.0,
    category_weights={
        "groceries": 0.25,
        "dining": 0.15,
        # ... other categories
    },
    avg_daily_transactions=3.0,
    spending_volatility=0.3,
    impulse_probability=0.1,
    peak_hours=[12, 19],
    weekend_multiplier=1.2,
)
```

## Technical Details

### Autoencoder Architecture

- **Input**: 15-dimensional feature vector
  - Amount (z-scored): 1 dim
  - Category (one-hot): 9 dims
  - Hour (sin/cos): 2 dims
  - Day of week (sin/cos): 2 dims
  - Days to month end (scaled): 1 dim

- **Architecture**: Dense 15 → 32 → 8 → 32 → 15
- **Activation**: ReLU
- **Loss**: MSE
- **Optimizer**: Adam
- **Training**: 20-30 epochs with early stopping

### Anomaly Detection

- Threshold set at 95th percentile of validation reconstruction errors
- Transaction flagged as anomaly if `recon_error > threshold`
- Error ratio (`recon_error / threshold`) used for severity scoring

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

This project was built as an MVP demonstration of AI-powered financial coaching using anomaly detection and contextual nudging.
