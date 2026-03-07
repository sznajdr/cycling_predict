# Contributing to Cycling Predict

Thank you for contributing! This guide will help you get set up and follow our workflow.

## Quick Start for Team Members

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/cycling-predict.git
cd cycling-predict
```

### 2. Install Dependencies

```bash
# Install the procyclingstats library (required for scraping)
pip install -e ../procyclingstats

# Install all other dependencies
pip install -r requirements.txt
```

### 3. Verify Setup

```bash
# Run tests
python tests/test_connection.py
python tests/test_rider.py
python tests/test_race.py

# Run quick demo
python quickstart.py
```

## Development Workflow

### Branch Naming

- `feature/strategy-X-description` - New strategy implementation
- `bugfix/description` - Bug fixes
- `docs/description` - Documentation updates
- `data/race-name-year` - Adding new race data

### Making Changes

1. **Create a branch:**
```bash
git checkout -b feature/strategy-3-medical-pk
```

2. **Make your changes**

3. **Test locally:**
```bash
pytest tests/betting/test_strategies.py -v
python quickstart.py
```

4. **Commit with clear messages:**
```bash
git add .
git commit -m "Strategy 3: Implement Medical PK model

- Two-compartment pharmacokinetic model
- Robust Kelly sizing with posterior uncertainty
- SQL schema for pk_parameters table"
```

5. **Push and create Pull Request:**
```bash
git push origin feature/strategy-3-medical-pk
```

## Code Style

- Follow PEP 8
- Use type hints where possible
- Add docstrings to all public functions
- Keep functions focused and small

### Example:

```python
def calculate_kelly_fraction(
    prob: float, 
    odds: float,
    prob_std: Optional[float] = None
) -> float:
    """
    Calculate Kelly-optimal bet fraction.
    
    Args:
        prob: Estimated win probability
        odds: Decimal odds from market
        prob_std: Standard deviation of probability (for robust Kelly)
    
    Returns:
        Optimal fraction of bankroll to bet
    """
    # Implementation here
    pass
```

## Database Changes

When modifying the database schema:

1. Add changes to `genqirue/data/schema_extensions.sql`
2. Create migration scripts in `migrations/`
3. Test on a copy of the database first

## Adding New Strategies

See `PLAN.md` for the 15 strategies. To add a new one:

1. Create file: `genqirue/models/strategy_name.py`
2. Inherit from `BayesianModel` or appropriate base class
3. Implement required methods: `build_model()`, `fit()`, `predict()`, `get_edge()`
4. Add SQL schema if needed
5. Add tests in `tests/betting/test_strategies.py`
6. Update `genqirue/models/__init__.py`
7. Update `genqirue/README.md`

### Strategy Template:

```python
"""
Strategy X: Description
"""
from typing import Dict, Any
from genqirue.models.base import BayesianModel, StrategyMixin

class NewStrategyModel(BayesianModel, StrategyMixin):
    """Description of the strategy."""
    
    def __init__(self, model_name="new_strategy"):
        super().__init__(model_name=model_name)
    
    def build_model(self, data: Dict[str, Any]):
        """Build PyMC model."""
        pass
    
    def predict(self, new_data: Dict[str, Any]):
        """Generate predictions."""
        pass
    
    def get_edge(self, prediction: Dict[str, Any], market_odds: float) -> float:
        """Calculate betting edge in basis points."""
        pass
```

## Testing

Run all tests before submitting:

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/betting/test_strategies.py -v

# With coverage
pytest tests/ --cov=genqirue --cov-report=html
```

## Data Sharing

The SQLite database (`data/cycling.db`) is **not** in git (it's in `.gitignore`).

To share data with the team:

1. **Export specific races:**
```bash
python scripts/export_race_data.py --race tour-de-france --year 2024
```

2. **Share via cloud storage** (Google Drive, Dropbox, etc.)

3. **Or use a shared database** (PostgreSQL, ClickHouse - see deployment guide)

## Environment Setup

Use a virtual environment:

```bash
# Create
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Questions?

- Check `README.md` for basic usage
- Check `SCRAPE_README.md` for scraping details
- Check `genqirue/README.md` for model documentation
- Open an issue on GitHub

## Code Review Checklist

Before submitting a PR:

- [ ] Tests pass
- [ ] Code follows style guide
- [ ] Docstrings added
- [ ] Type hints included
- [ ] SQL schema updated if needed
- [ ] README updated if needed
- [ ] No large binary files added
- [ ] .gitignore updated if needed

## Deployment

For production deployment, see `DEPLOYMENT.md`.

---

Happy coding! 🚴‍♂️📊
