-- Database schema extensions for Genqirue betting engine.
-- 
-- This SQL adds tables for:
-- - Strategy outputs and model predictions
-- - Rider frailty and tactical states  
-- - Weather predictions
-- - Betting positions and portfolio tracking
-- - Performance attribution
--
-- Run this after the base cycling.db schema is created.

-- ============================================================
-- STRATEGY 2: Gruppetto Frailty Model
-- ============================================================

CREATE TABLE IF NOT EXISTS rider_frailty (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rider_id INTEGER NOT NULL,
    frailty_estimate REAL NOT NULL,
    hidden_form_prob REAL NOT NULL DEFAULT 0.0,
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_version TEXT,
    FOREIGN KEY (rider_id) REFERENCES riders(id),
    UNIQUE(rider_id, computed_at)
);

CREATE INDEX IF NOT EXISTS idx_frailty_rider ON rider_frailty(rider_id);
CREATE INDEX IF NOT EXISTS idx_frailty_hidden_form ON rider_frailty(hidden_form_prob) 
    WHERE hidden_form_prob > 0.3;

-- View for hidden form detection (as specified in PLAN.md)
CREATE VIEW IF NOT EXISTS rider_hidden_form AS
SELECT 
    rf.rider_id,
    r.name as rider_name,
    rf.frailty_estimate,
    rf.hidden_form_prob,
    rf.computed_at,
    CASE 
        WHEN rf.frailty_estimate > (
            SELECT AVG(frailty_estimate) + 1.5 * STDDEV(frailty_estimate) 
            FROM rider_frailty 
            WHERE computed_at = (SELECT MAX(computed_at) FROM rider_frailty)
        )
        THEN 1 - (
            SELECT CUME_DIST() OVER (ORDER BY frailty_estimate)
            FROM rider_frailty rf2
            WHERE rf2.rider_id = rf.rider_id
            AND rf2.computed_at = rf.computed_at
        )
        ELSE 0 
    END as hidden_form_signal
FROM rider_frailty rf
JOIN riders r ON rf.rider_id = r.id
WHERE rf.computed_at = (SELECT MAX(computed_at) FROM rider_frailty);

-- ============================================================
-- STRATEGY 1: Tactical Time Loss HMM
-- ============================================================

CREATE TABLE IF NOT EXISTS tactical_states (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rider_id INTEGER NOT NULL,
    stage_id INTEGER NOT NULL,
    contesting_prob REAL NOT NULL,
    preserving_prob REAL NOT NULL,
    recovering_prob REAL DEFAULT 0.0,
    gruppetto_prob REAL DEFAULT 0.0,
    decoded_state TEXT NOT NULL,
    tactical_time_loss_seconds REAL,
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (rider_id) REFERENCES riders(id),
    FOREIGN KEY (stage_id) REFERENCES race_stages(id),
    UNIQUE(rider_id, stage_id)
);

CREATE INDEX IF NOT EXISTS idx_tactical_rider ON tactical_states(rider_id);
CREATE INDEX IF NOT EXISTS idx_tactical_stage ON tactical_states(stage_id);
CREATE INDEX IF NOT EXISTS idx_tactical_preserving ON tactical_states(preserving_prob) 
    WHERE preserving_prob > 0.7;

-- ============================================================
-- STRATEGY 6: Weather SPDE Model
-- ============================================================

CREATE TABLE IF NOT EXISTS weather_fields (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stage_id INTEGER NOT NULL,
    distance_km REAL NOT NULL,
    time_minutes REAL NOT NULL,
    wind_speed_pred REAL NOT NULL,
    wind_speed_std REAL,
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (stage_id) REFERENCES race_stages(id),
    UNIQUE(stage_id, distance_km, time_minutes)
);

CREATE INDEX IF NOT EXISTS idx_weather_stage ON weather_fields(stage_id);

CREATE TABLE IF NOT EXISTS itt_time_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stage_id INTEGER NOT NULL,
    rider_early_id INTEGER NOT NULL,
    rider_late_id INTEGER NOT NULL,
    delta_t_seconds REAL NOT NULL,
    uncertainty REAL,
    favors_rider TEXT,
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (stage_id) REFERENCES race_stages(id),
    FOREIGN KEY (rider_early_id) REFERENCES riders(id),
    FOREIGN KEY (rider_late_id) REFERENCES riders(id),
    UNIQUE(stage_id, rider_early_id, rider_late_id)
);

-- ============================================================
-- STRATEGY 12: Online Changepoint Detection
-- ============================================================

CREATE TABLE IF NOT EXISTS telemetry_changepoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rider_id INTEGER NOT NULL,
    stage_id INTEGER NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    run_length INTEGER NOT NULL,
    changepoint_prob REAL NOT NULL,
    power_z_score REAL,
    attack_signal INTEGER DEFAULT 0,  -- 0=none, 1=suspected, 2=confirmed, 3=established
    latency_ms REAL,
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (rider_id) REFERENCES riders(id),
    FOREIGN KEY (stage_id) REFERENCES race_stages(id)
);

CREATE INDEX IF NOT EXISTS idx_changepoint_rider ON telemetry_changepoints(rider_id);
CREATE INDEX IF NOT EXISTS idx_changepoint_stage ON telemetry_changepoints(stage_id);
CREATE INDEX IF NOT EXISTS idx_changepoint_signal ON telemetry_changepoints(attack_signal) 
    WHERE attack_signal >= 2;

-- ============================================================
-- STRATEGY 3: Medical PK Model
-- ============================================================

CREATE TABLE IF NOT EXISTS pk_parameters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rider_id INTEGER NOT NULL,
    incident_date DATE NOT NULL,
    incident_type TEXT,  -- 'crash', 'illness', 'mechanical'
    k_elimination REAL,  -- Elimination rate constant
    ec50 REAL,           -- EC50 for performance effect
    recovery_factor REAL DEFAULT 1.0,
    estimated_full_recovery_date DATE,
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (rider_id) REFERENCES riders(id),
    UNIQUE(rider_id, incident_date)
);

CREATE INDEX IF NOT EXISTS idx_pk_rider ON pk_parameters(rider_id);

-- ============================================================
-- GENERAL: Model Predictions & Strategy Outputs
-- ============================================================

CREATE TABLE IF NOT EXISTS strategy_outputs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_name TEXT NOT NULL,
    rider_id INTEGER NOT NULL,
    stage_id INTEGER NOT NULL,
    win_prob REAL NOT NULL,
    win_prob_std REAL DEFAULT 0.0,
    edge_bps REAL DEFAULT 0.0,
    expected_value REAL DEFAULT 0.0,
    r_hat REAL DEFAULT 1.0,
    ess REAL DEFAULT 0.0,
    latent_states_json TEXT,  -- JSON-encoded latent variables
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (rider_id) REFERENCES riders(id),
    FOREIGN KEY (stage_id) REFERENCES race_stages(id),
    UNIQUE(strategy_name, rider_id, stage_id, computed_at)
);

CREATE INDEX IF NOT EXISTS idx_strategy_name ON strategy_outputs(strategy_name);
CREATE INDEX IF NOT EXISTS idx_strategy_rider ON strategy_outputs(rider_id);
CREATE INDEX IF NOT EXISTS idx_strategy_stage ON strategy_outputs(stage_id);
CREATE INDEX IF NOT EXISTS idx_strategy_edge ON strategy_outputs(edge_bps) 
    WHERE edge_bps > 50;

-- ============================================================
-- PORTFOLIO: Positions and Betting History
-- ============================================================

CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_name TEXT NOT NULL,
    rider_id INTEGER NOT NULL,
    stage_id INTEGER NOT NULL,
    market_type TEXT NOT NULL,  -- 'winner', 'top3', 'h2h', etc.
    
    -- Kelly sizing
    kelly_fraction REAL DEFAULT 0.0,
    robust_kelly_fraction REAL DEFAULT 0.0,
    half_kelly_fraction REAL DEFAULT 0.0,
    
    -- Risk metrics
    cvar_95 REAL DEFAULT 0.0,
    max_drawdown REAL DEFAULT 0.0,
    
    -- Execution
    stake_units REAL DEFAULT 0.0,  -- Final stake after all constraints
    entry_odds REAL,
    exit_odds REAL,
    
    -- P&L tracking
    status TEXT DEFAULT 'pending',  -- pending, open, closed, void
    pnl_units REAL,
    
    -- Metadata
    confidence TEXT,  -- 'extreme', 'high', 'medium', 'low', 'speculative'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    closed_at TIMESTAMP,
    
    FOREIGN KEY (rider_id) REFERENCES riders(id),
    FOREIGN KEY (stage_id) REFERENCES race_stages(id)
);

CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
CREATE INDEX IF NOT EXISTS idx_positions_strategy ON positions(strategy_name);
CREATE INDEX IF NOT EXISTS idx_positions_rider ON positions(rider_id);

-- Portfolio summary view
CREATE VIEW IF NOT EXISTS portfolio_summary AS
SELECT 
    strategy_name,
    COUNT(*) as total_positions,
    SUM(CASE WHEN status = 'open' THEN 1 ELSE 0 END) as open_positions,
    SUM(CASE WHEN status = 'closed' THEN 1 ELSE 0 END) as closed_positions,
    SUM(stake_units) as total_staked,
    SUM(pnl_units) as total_pnl,
    AVG(pnl_units) as avg_pnl,
    SUM(CASE WHEN pnl_units > 0 THEN 1 ELSE 0 END) * 1.0 / 
        NULLIF(SUM(CASE WHEN status = 'closed' THEN 1 ELSE 0 END), 0) as win_rate
FROM positions
GROUP BY strategy_name;

-- ============================================================
-- VALIDATION: Backtest and Performance Tracking
-- ============================================================

CREATE TABLE IF NOT EXISTS backtest_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_name TEXT NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    n_predictions INTEGER,
    
    -- Scoring metrics
    brier_score REAL,
    crps REAL,  -- Continuous Ranked Probability Score
    rps REAL,   -- Ranked Probability Score
    
    -- Calibration
    calibration_slope REAL,
    calibration_intercept REAL,
    
    -- Financial
    total_return_pct REAL,
    sharpe_ratio REAL,
    max_drawdown_pct REAL,
    
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(strategy_name, start_date, end_date)
);

CREATE INDEX IF NOT EXISTS idx_backtest_strategy ON backtest_results(strategy_name);

-- ============================================================
-- SYSTEM: Model Versions and Freshness
-- ============================================================

CREATE TABLE IF NOT EXISTS model_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_name TEXT NOT NULL,
    version TEXT NOT NULL,
    git_commit_hash TEXT,
    fitted_at TIMESTAMP,
    hyperparameters_json TEXT,
    performance_summary_json TEXT,
    is_active BOOLEAN DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(strategy_name, version)
);

CREATE INDEX IF NOT EXISTS idx_model_versions ON model_versions(strategy_name, is_active);

-- ============================================================
-- ODDS: Bookmaker odds snapshots (Betclic + future sources)
-- ============================================================

CREATE TABLE IF NOT EXISTS bookmaker_odds (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    bookmaker               TEXT NOT NULL DEFAULT 'betclic',
    event_url               TEXT NOT NULL,
    event_id                TEXT NOT NULL,
    market_type             TEXT NOT NULL,
    market_label_raw        TEXT NOT NULL,
    participant_name        TEXT NOT NULL,
    participant_name_norm   TEXT,
    participant_raw         TEXT NOT NULL,
    back_odds               REAL NOT NULL,
    implied_prob            REAL NOT NULL,
    market_total_impl_prob  REAL,
    fair_prob               REAL,
    fair_odds               REAL,
    scraped_at              TEXT NOT NULL,
    scrape_run_id           TEXT NOT NULL,
    race_id                 INTEGER REFERENCES races(id),
    UNIQUE(bookmaker, event_id, market_type, participant_raw, scrape_run_id)
);

CREATE INDEX IF NOT EXISTS idx_bookmaker_odds_event
    ON bookmaker_odds(event_id, market_type, scraped_at);
CREATE INDEX IF NOT EXISTS idx_bookmaker_odds_participant
    ON bookmaker_odds(participant_name, market_type, scraped_at);

CREATE VIEW IF NOT EXISTS bookmaker_odds_latest AS
SELECT bo.* FROM bookmaker_odds bo
INNER JOIN (
    SELECT event_id, market_type, participant_raw, MAX(scraped_at) AS max_scraped
    FROM bookmaker_odds
    GROUP BY event_id, market_type, participant_raw
) latest ON bo.event_id = latest.event_id
         AND bo.market_type = latest.market_type
         AND bo.participant_raw = latest.participant_raw
         AND bo.scraped_at = latest.max_scraped;
