"""Basic tests for Financial Coach components."""

import os
import sys
import tempfile
from datetime import datetime

import pytest
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.personas import get_persona, list_personas, Persona
from src.data.generator import TransactionGenerator, generate_persona_dataset
from src.models.autoencoder import (
    TransactionAutoencoder, 
    FeatureProcessor, 
    AnomalyDetector
)
from src.host.policy import PolicyEngine, PolicyContext, Action


class TestPersonas:
    """Test persona definitions."""
    
    def test_list_personas(self):
        """Should return list of persona names."""
        personas = list_personas()
        assert len(personas) == 5
        assert "balanced_ben" in personas
    
    def test_get_persona(self):
        """Should return valid persona."""
        persona = get_persona("balanced_ben")
        assert isinstance(persona, Persona)
        assert persona.monthly_income > 0
        assert persona.savings_goal > 0
        assert sum(persona.category_weights.values()) == pytest.approx(1.0, rel=0.01)
    
    def test_invalid_persona(self):
        """Should raise error for invalid persona."""
        with pytest.raises(ValueError):
            get_persona("nonexistent")


class TestDataGenerator:
    """Test transaction data generation."""
    
    def test_generate_transactions(self):
        """Should generate valid transaction stream."""
        persona = get_persona("balanced_ben")
        generator = TransactionGenerator(persona=persona, seed=42)
        
        transactions = generator.generate_stream(
            start_date=datetime(2024, 1, 1),
            days=30,
        )
        
        assert len(transactions) > 0
        assert all(t.amount > 0 for t in transactions)
        assert all(t.category in persona.category_weights for t in transactions)
    
    def test_generate_dataset(self):
        """Should generate DataFrame with required columns."""
        df = generate_persona_dataset("balanced_ben", days=30, seed=42)
        
        required_cols = ["amount", "category", "hour", "day_of_week", "days_to_month_end"]
        for col in required_cols:
            assert col in df.columns
        
        assert len(df) > 0
        assert df["amount"].min() > 0
    
    def test_reproducibility(self):
        """Same seed should produce same data."""
        # Use fixed start date for reproducibility
        fixed_start = datetime(2024, 1, 1)
        df1 = generate_persona_dataset("balanced_ben", days=30, seed=42, start_date=fixed_start)
        df2 = generate_persona_dataset("balanced_ben", days=30, seed=42, start_date=fixed_start)
        
        pd.testing.assert_frame_equal(df1, df2)


class TestAutoencoder:
    """Test autoencoder model."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample training data."""
        return generate_persona_dataset("balanced_ben", days=60, seed=42)
    
    def test_feature_processor(self, sample_data):
        """Should transform data to correct dimensions."""
        processor = FeatureProcessor()
        X = processor.fit_transform(sample_data)
        
        assert X.shape[0] == len(sample_data)
        assert X.shape[1] == 15  # Expected feature dimensions
        assert not np.isnan(X).any()
    
    def test_model_forward(self):
        """Model should produce output of same shape as input."""
        import torch
        
        model = TransactionAutoencoder(input_dim=15)
        x = torch.randn(10, 15)
        output = model(x)
        
        assert output.shape == x.shape
    
    def test_anomaly_detector_train(self, sample_data):
        """Should train and produce valid threshold."""
        detector = AnomalyDetector(persona_name="test")
        metrics = detector.train(sample_data, epochs=5, verbose=False)
        
        assert detector.trained
        assert detector.threshold > 0
        assert metrics["epochs_trained"] > 0
    
    def test_anomaly_detector_score(self, sample_data):
        """Should score transactions and detect anomalies."""
        detector = AnomalyDetector(persona_name="test")
        detector.train(sample_data, epochs=5, verbose=False)
        
        errors, is_anomaly = detector.score(sample_data)
        
        assert len(errors) == len(sample_data)
        assert len(is_anomaly) == len(sample_data)
        assert all(e >= 0 for e in errors)
    
    def test_save_load(self, sample_data):
        """Should save and load model correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Train and save
            detector = AnomalyDetector(persona_name="test")
            detector.train(sample_data, epochs=5, verbose=False)
            detector.save(tmpdir)
            
            # Load and compare
            loaded = AnomalyDetector.load(tmpdir)
            
            assert loaded.threshold == detector.threshold
            assert loaded.trained


class TestPolicy:
    """Test policy engine."""
    
    def test_do_nothing_when_on_track(self):
        """Should do nothing for normal transactions when on track."""
        policy = PolicyEngine()
        
        context = PolicyContext(
            amount=50.0,
            category="groceries",
            is_anomaly=False,
            recon_error=0.5,
            threshold=1.0,
            monthly_budget=4000.0,
            spent_this_month=1500.0,
            savings_goal=500.0,
            days_to_month_end=15,
        )
        
        action, _ = policy.decide(context)
        assert action == Action.DO_NOTHING
    
    def test_gentle_nudge_for_anomaly(self):
        """Should nudge for anomalous transactions."""
        policy = PolicyEngine()
        
        context = PolicyContext(
            amount=200.0,
            category="entertainment",
            is_anomaly=True,
            recon_error=1.5,
            threshold=1.0,
            monthly_budget=4000.0,
            spent_this_month=2000.0,
            savings_goal=500.0,
            days_to_month_end=15,
        )
        
        action, _ = policy.decide(context)
        assert action in [Action.GENTLE_NUDGE, Action.GOAL_REMINDER]
    
    def test_goal_reminder_when_over_budget(self):
        """Should send goal reminder when significantly over budget."""
        policy = PolicyEngine()
        
        context = PolicyContext(
            amount=300.0,
            category="shopping",
            is_anomaly=True,
            recon_error=2.0,
            threshold=1.0,
            monthly_budget=4000.0,
            spent_this_month=3800.0,  # Almost at budget
            savings_goal=500.0,
            days_to_month_end=5,  # End of month
        )
        
        action, _ = policy.decide(context)
        assert action == Action.GOAL_REMINDER


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

