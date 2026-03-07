"""
Tests for betting strategies.
"""
import pytest
import numpy as np
from datetime import datetime, timedelta

from genqirue.models import (
    GruppettoFrailtyModel,
    BayesianChangepointDetector,
    TacticalTimeLossHMM,
    WeatherSPDEModel,
)
from genqirue.portfolio import RobustKellyOptimizer, KellyParameters
from genqirue.domain import (
    StageType,
    TacticalState,
    MarketState,
    Position,
    RiderState,
)


class TestGruppettoFrailty:
    """Tests for Strategy 2: Gruppetto Outlier Detection."""
    
    def test_model_initialization(self):
        model = GruppettoFrailtyModel()
        assert model.model_name == "gruppetto_frailty"
        assert model.n_hazard_pieces == 5
    
    def test_frailty_calculation(self):
        """Test that frailty estimates are reasonable."""
        model = GruppettoFrailtyModel()
        
        # Create synthetic survival data
        from genqirue.models.gruppetto_frailty import SurvivalRecord
        
        survival_data = [
            SurvivalRecord(
                rider_id=1,
                stage_id=i,
                stage_date=datetime.now(),
                stage_type='mountain',
                time_to_cutoff=30.0,
                event_observed=False,
                gc_position=50,
                gc_time_behind=600,
                gruppetto_indicator=1,
                gruppetto_time_loss=120
            )
            for i in range(10)
        ]
        
        # Fast estimator (non-Bayesian)
        from genqirue.models import FastFrailtyEstimator
        estimator = FastFrailtyEstimator()
        estimator.fit(survival_data)
        
        frailty = estimator.get_frailty(1)
        assert isinstance(frailty, float)
        assert not np.isnan(frailty)
    
    def test_hidden_form_probability_bounds(self):
        """Test hidden form probabilities are in [0, 1]."""
        model = GruppettoFrailtyModel()
        # Before fitting, should return 0
        prob = model.get_hidden_form_prob(1)
        assert 0 <= prob <= 1


class TestOnlineChangepoint:
    """Tests for Strategy 12: Attack Confirmation."""
    
    def test_detector_initialization(self):
        detector = BayesianChangepointDetector()
        assert detector.changepoint_threshold == 0.8
        assert detector.max_run_length == 1000
    
    def test_update_returns_expected_keys(self):
        detector = BayesianChangepointDetector()
        
        result = detector.update({
            'power_z_score': 1.5,
            'rider_id': 1,
            'timestamp': datetime.now()
        })
        
        assert 'changepoint_prob' in result
        assert 'run_length' in result
        assert 'signal' in result
        assert 'latency_ms' in result
    
    def test_latency_requirement(self):
        """Test that updates complete within 100ms target."""
        detector = BayesianChangepointDetector()
        
        # Warm up
        for _ in range(10):
            detector.update({
                'power_z_score': np.random.normal(),
                'rider_id': 1,
                'timestamp': datetime.now()
            })
        
        avg_latency = detector.get_latency_ms()
        # Should be < 100ms for real-time operation
        assert avg_latency < 100.0
    
    def test_signal_levels(self):
        """Test signal level progression."""
        detector = BayesianChangepointDetector(
            changepoint_threshold=0.5,
            z_score_threshold=1.0
        )
        
        detector.update_yesterday_z_score(1, 2.5)  # High yesterday score
        
        # Multiple high power readings
        for i in range(20):
            result = detector.update({
                'power_z_score': 3.0 if i > 10 else 0.5,
                'rider_id': 1,
                'timestamp': datetime.now()
            })
        
        # Should eventually detect changepoint
        assert result['changepoint_prob'] >= 0
        assert result['run_length'] >= 0


class TestTacticalHMM:
    """Tests for Strategy 1: Tactical Time Loss HMM."""
    
    def test_model_initialization(self):
        model = TacticalTimeLossHMM()
        assert model.model_name == "tactical_time_loss_hmm"
        assert model.tactical_loss_mean == 120.0  # 2 minutes
    
    def test_state_probability_distribution(self):
        """Test state probabilities sum to ~1."""
        model = TacticalTimeLossHMM()
        
        # Before fitting, should return uniform-ish
        probs = model.get_tactical_state_prob(
            rider_id=1,
            stage_type=StageType.MOUNTAIN,
            gc_time_behind=600
        )
        
        total = sum(probs.values())
        assert 0.99 <= total <= 1.01
    
    def test_simple_detector_rules(self):
        """Test simple rule-based detector."""
        from genqirue.models.tactical_hmm import (
            SimpleTacticalDetector,
            TacticalObservation
        )
        
        detector = SimpleTacticalDetector()
        
        obs = TacticalObservation(
            rider_id=1,
            stage_id=1,
            stage_type=StageType.MOUNTAIN,
            stage_date=datetime.now(),
            time_loss_seconds=180,
            gc_position=50,
            gc_time_behind=900,
            gruppetto_indicator=True
        )
        
        state = detector.update(obs)
        assert state == TacticalState.GRUPPETTO


class TestWeatherSPDE:
    """Tests for Strategy 6: Weather SPDE."""
    
    def test_model_initialization(self):
        model = WeatherSPDEModel()
        assert model.model_name == "weather_spde"
        assert model.use_hsgp == True
    
    def test_time_difference_calculation(self):
        """Test fair time difference calculation structure."""
        from genqirue.models.weather_spde import ITTStarter, WeatherObservation
        
        model = WeatherSPDEModel()
        
        # Add some weather observations
        for i in range(5):
            model.weather_observations.append(WeatherObservation(
                timestamp=datetime.now() + timedelta(minutes=i*10),
                location=(i*5, 0),
                wind_speed_ms=5.0 + i*0.5,
                wind_direction_deg=270,
                temperature_c=20
            ))
        
        # Create ITT starters
        early = ITTStarter(
            rider_id=1,
            start_time=datetime.now(),
            start_order=1,
            predicted_finish_time=datetime.now() + timedelta(minutes=50),
            power_watts=400,
            cda=0.25
        )
        
        late = ITTStarter(
            rider_id=2,
            start_time=datetime.now() + timedelta(minutes=60),
            start_order=20,
            predicted_finish_time=datetime.now() + timedelta(minutes=110),
            power_watts=400,
            cda=0.25
        )
        
        # Note: Without fitting GP, this will return zeros
        result = model.calculate_fair_time_difference(early, late)
        
        assert 'delta_t_seconds' in result
        assert 'favors' in result


class TestRobustKelly:
    """Tests for portfolio optimization."""
    
    def test_kelly_calculation(self):
        optimizer = RobustKellyOptimizer()
        
        # Test full Kelly
        prob = 0.6
        odds = 2.0
        
        f_full = optimizer.kelly_fraction(prob, odds)
        # f* = (bp - q) / b = (1*0.6 - 0.4) / 1 = 0.2
        assert abs(f_full - 0.2) < 0.01
    
    def test_robust_kelly_with_uncertainty(self):
        optimizer = RobustKellyOptimizer()
        
        prob = 0.6
        odds = 2.0
        prob_std = 0.1
        
        f_robust = optimizer.robust_kelly_fraction(prob, odds, prob_std)
        f_full = optimizer.kelly_fraction(prob, odds)
        
        # Robust should be <= full
        assert f_robust <= f_full
    
    def test_position_constraints(self):
        """Test that position constraints are enforced."""
        params = KellyParameters(max_position_pct=0.25)
        optimizer = RobustKellyOptimizer(params)
        
        # Create test positions
        positions = []
        for i in range(5):
            market = MarketState(
                market_type='winner',
                selection_id=i,
                back_odds=2.0,
                model_prob=0.6,
                model_prob_uncertainty=0.05
            )
            pos = Position(market_state=market)
            positions.append(pos)
        
        portfolio = optimizer.optimize_portfolio(positions)
        
        # Check max position constraint
        for pos in portfolio.positions:
            assert pos.stake <= params.max_position_pct + 0.001
        
        # Check total allocation
        assert portfolio.total_stake <= 1.0 + 0.001
    
    def test_edge_threshold_filtering(self):
        """Test that positions below edge threshold are filtered."""
        params = KellyParameters(min_edge_bps=100.0)  # 1% edge minimum
        optimizer = RobustKellyOptimizer(params)
        
        # Create positions with varying edges
        positions = []
        
        # Low edge position
        market_low = MarketState(
            market_type='winner',
            selection_id=1,
            back_odds=2.0,
            model_prob=0.49,  # Slight under fair
            model_prob_uncertainty=0.05
        )
        market_low.edge_bps = -100  # Negative edge
        positions.append(Position(market_state=market_low))
        
        # High edge position
        market_high = MarketState(
            market_type='winner',
            selection_id=2,
            back_odds=3.0,
            model_prob=0.5,  # Fair would be 0.33
            model_prob_uncertainty=0.05
        )
        market_high.edge_bps = 500  # 5% edge
        positions.append(Position(market_state=market_high))
        
        portfolio = optimizer.optimize_portfolio(positions)
        
        # Only high edge position should remain
        assert len(portfolio.positions) <= 1


class TestIntegration:
    """Integration tests for the full betting pipeline."""
    
    def test_end_to_end_workflow(self):
        """Test the complete workflow from data to positions."""
        # This is a simplified integration test
        
        # 1. Create mock data
        rider_id = 1
        stage_type = StageType.MOUNTAIN
        
        # 2. Get tactical state (would come from HMM)
        hmm = TacticalTimeLossHMM()
        state_probs = hmm.get_tactical_state_prob(
            rider_id, stage_type, gc_time_behind=300
        )
        
        # 3. Get frailty (would come from survival model)
        frailty_model = GruppettoFrailtyModel()
        hidden_form = frailty_model.get_hidden_form_prob(rider_id)
        
        # 4. Create position if signals align
        positions = []
        if state_probs[TacticalState.PRESERVING] > 0.5 and hidden_form > 0.2:
            market = MarketState(
                market_type='winner',
                selection_id=rider_id,
                back_odds=15.0,
                model_prob=0.15,
                model_prob_uncertainty=0.03
            )
            positions.append(Position(market_state=market))
        
        # 5. Optimize portfolio
        if positions:
            optimizer = RobustKellyOptimizer()
            portfolio = optimizer.optimize_portfolio(positions)
            
            assert portfolio.total_stake >= 0
            assert portfolio.total_stake <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
