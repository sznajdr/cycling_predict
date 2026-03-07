# Contributing

Guidelines for contributing code, data, and documentation to this project.

---

## Table of Contents

1. [Getting Started](#1-getting-started)
2. [Repository Setup](#2-repository-setup)
3. [Daily Workflow](#3-daily-workflow)
4. [Adding a New Strategy](#4-adding-a-new-strategy)
5. [Code Standards](#5-code-standards)
6. [Testing Requirements](#6-testing-requirements)
7. [Database Changes](#7-database-changes)
8. [Pull Request Checklist](#8-pull-request-checklist)
9. [GitHub Administration](#9-github-administration)

---

## 1. Getting Started

### Clone and install

```bash
git clone https://github.com/your-org/cycling-predict.git
cd cycling-predict

pip install -e ../procyclingstats   # sibling repo — required for scraping
pip install -r requirements.txt

python fetch_odds.py --init-schema  # apply all DB schemas
```

### Verify the setup

```bash
python tests/test_connection.py     # PCS connectivity
python tests/test_rider.py          # scrape + DB roundtrip
python tests/test_race.py           # race meta roundtrip
python quickstart.py                # end-to-end demo (requires scraped data)
```

All tests should print `PASS`. If `test_connection.py` fails, PCS is rate-limiting — wait a few minutes and retry.

---

## 2. Repository Setup

### Creating the remote (first time only)

```bash
# Create a private repository on GitHub, then:
git remote add origin https://github.com/your-org/cycling-predict.git
git branch -M main
git push -u origin main
```

### Authentication

Use a Personal Access Token (PAT) rather than a password:

1. GitHub → Settings → Developer settings → Personal access tokens → Generate new token (classic)
2. Scopes: select `repo`
3. Use the token as the password when prompted by git

Or use the GitHub CLI:

```bash
gh auth login
```

### Branch protection (recommended)

In the repository settings under Branches, add a rule for `main`:

- Require a pull request before merging
- Require at least one approval
- Require status checks to pass before merging
- Require conversation resolution before merging

---

## 3. Daily Workflow

```bash
# 1. Pull latest main
git pull origin main

# 2. Create a branch
git checkout -b feature/strategy-3-medical-pk

# 3. Make changes

# 4. Test
pytest tests/betting/test_strategies.py -v
python quickstart.py

# 5. Stage only the files you changed
git add genqirue/models/medical_pk.py tests/betting/test_strategies.py

# 6. Commit with a clear message
git commit -m "Strategy 3: two-compartment PK model for trauma recovery

- Implements dC_trauma/dt = -k_el * C_trauma with EC50 performance effect
- Robust Kelly sizing with posterior uncertainty
- Unit tests for parameter estimation and edge calculation"

# 7. Push and open a PR
git push origin feature/strategy-3-medical-pk
```

### Branch naming

| Prefix | Use for |
|--------|---------|
| `feature/` | New strategy or feature |
| `bugfix/` | Bug fix |
| `docs/` | Documentation changes only |
| `data/` | Race config changes |

---

## 4. Adding a New Strategy

All 15 strategies are mathematically specified in `docs/MODELS.md`. To implement one:

1. Create `genqirue/models/<strategy_name>.py` — inherit from `BayesianModel` in `base.py`
2. Implement required methods: `build_model()`, `fit()`, `predict()`, `get_edge()`
3. Add SQL schema changes to `genqirue/data/schema_extensions.sql` if new tables are needed
4. Add tests in `tests/betting/test_strategies.py`
5. Export the class from `genqirue/models/__init__.py`
6. Update the status table in `docs/ENGINE.md`

### Strategy template

```python
"""
Strategy N: Description

Targets: <the market mispricing>
Model:   <the mathematical approach>
"""
from typing import Dict, Any
from genqirue.models.base import BayesianModel, StrategyMixin


class NewStrategyModel(BayesianModel, StrategyMixin):
    """One-line description."""

    def __init__(self, model_name: str = "new_strategy"):
        super().__init__(model_name=model_name)

    def build_model(self, data: Dict[str, Any]):
        """Build the PyMC model."""
        pass

    def fit(self, data: Dict[str, Any]):
        """Fit the model to historical data."""
        pass

    def predict(self, new_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate win probability predictions with uncertainty."""
        pass

    def get_edge(self, prediction: Dict[str, Any], market_odds: float) -> float:
        """Return edge in basis points. Positive = value bet."""
        pass
```

---

## 5. Code Standards

- Follow PEP 8
- Type hints on all public function signatures
- Docstrings on all public classes and functions
- Functions should be short and do one thing
- No hardcoded paths — use `Path(__file__).parent` or arguments
- No personal names, local paths, or credentials in any committed file

### Example

```python
def calculate_kelly_fraction(
    prob: float,
    odds: float,
    prob_std: float | None = None,
) -> float:
    """
    Calculate Kelly-optimal bet fraction with optional uncertainty penalty.

    Args:
        prob:     Estimated win probability.
        odds:     Decimal odds from market.
        prob_std: Posterior std of probability estimate (for robust Kelly).

    Returns:
        Optimal fraction of bankroll to bet. Clipped to [0, 0.25].
    """
    b = odds - 1
    f = (b * prob - (1 - prob)) / b
    if prob_std is not None:
        f *= 1 - prob_std**2 * (b + 1)**2 / (prob**2 * (b + 1)**2)
    return max(0.0, min(f, 0.25))
```

---

## 6. Testing Requirements

- All new strategies need unit tests in `tests/betting/test_strategies.py`
- Unit tests must run without network access (mock HTTP calls)
- Network tests (`tests/test_connection.py`, `tests/test_rider.py`, `tests/test_race.py`) are integration tests — do not add more network calls to the unit test suite
- CI runs `pytest tests/betting/` on every push (see `.github/workflows/tests.yml`)

```bash
# Run unit tests only (CI equivalent)
pytest tests/betting/ -v

# Run with coverage
pytest tests/ --cov=genqirue --cov-report=html
```

---

## 7. Database Changes

When modifying the database schema:

1. Add `CREATE TABLE IF NOT EXISTS` or `ALTER TABLE` statements to `genqirue/data/schema_extensions.sql`
2. Use `IF NOT EXISTS` throughout — the schema is applied idempotently via `python fetch_odds.py --init-schema`
3. For column additions to existing tables, add them to the `_migrate_db()` function in `pipeline/db.py`
4. Test on a copy of your database before committing: `cp data/cycling.db data/cycling_backup.db`

---

## 8. Pull Request Checklist

Before submitting:

- [ ] Tests pass: `pytest tests/ -v`
- [ ] Code follows style guide
- [ ] Docstrings on all new public functions
- [ ] Type hints on all new function signatures
- [ ] SQL schema updated if new tables needed
- [ ] `docs/ENGINE.md` status table updated if implementing a strategy
- [ ] No personal names, local paths, or credentials committed
- [ ] `data/cycling.db` not staged (it is in `.gitignore`)
- [ ] Large binary files not staged

---

## 9. GitHub Administration

### Adding a collaborator

Repository → Settings → Collaborators → Add people → enter username or email.

The invited collaborator clones the repo and runs:

```bash
git clone https://github.com/your-org/cycling-predict.git
cd cycling-predict
python scripts/setup_team.py
```

### Sharing data

The database (`data/cycling.db`) is not in git — it is too large and is regenerated by the scraper. To share data:

**Option 1 — Export a specific race:**
```bash
python scripts/export_race_data.py --race tour-de-france --year 2024
# Creates tour-de-france_2024.zip — share via cloud storage
# Teammate imports:
python scripts/export_race_data.py --import-zip tour-de-france_2024.zip
```

**Option 2 — Shared cloud database:** See `docs/DEPLOYMENT.md` for PostgreSQL setup on Railway, Supabase, or AWS RDS.

**Option 3 — Manual sync:** Upload `data/cycling.db` to shared cloud storage; download and replace locally.

### Creating issues

Track strategy implementation and data work in GitHub Issues:

- `Implement Strategy 3: Medical PK model`
- `Add Tour de France 2025 to config/races.yaml`
- `Calibrate frailty model on 2024 data`
