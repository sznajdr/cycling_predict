"""
Team Setup Script
================

Run this script to set up the project for a new team member.

Usage:
    python scripts/setup_team.py

What it does:
1. Checks Python version
2. Creates virtual environment
3. Installs dependencies
4. Applies database schema
5. Runs tests to verify setup
6. Creates sample .env file
"""
import os
import sys
import subprocess
import venv
from pathlib import Path


def print_step(step_num, message):
    """Print formatted step message."""
    print(f"\n{'='*60}")
    print(f"Step {step_num}: {message}")
    print('='*60)


def check_python():
    """Check Python version."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print(f"❌ Python 3.10+ required, found {version.major}.{version.minor}")
        print("Download from: https://www.python.org/downloads/")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True


def create_venv():
    """Create virtual environment."""
    venv_path = Path('venv')
    
    if venv_path.exists():
        print("Virtual environment already exists")
        return True
    
    print("Creating virtual environment...")
    try:
        venv.create('venv', with_pip=True)
        print("✓ Virtual environment created")
        return True
    except Exception as e:
        print(f"❌ Failed to create venv: {e}")
        return False


def get_python_executable():
    """Get the Python executable path."""
    if os.name == 'nt':  # Windows
        return 'venv\\Scripts\\python.exe'
    else:  # Mac/Linux
        return 'venv/bin/python'


def get_pip_executable():
    """Get the pip executable path."""
    if os.name == 'nt':  # Windows
        return 'venv\\Scripts\\pip.exe'
    else:  # Mac/Linux
        return 'venv/bin/pip'


def install_dependencies():
    """Install Python dependencies."""
    pip = get_pip_executable()
    
    print("Installing dependencies from requirements.txt...")
    result = subprocess.run(
        [pip, 'install', '-r', 'requirements.txt'],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"❌ Failed to install dependencies:")
        print(result.stderr)
        return False
    
    print("✓ Dependencies installed")
    return True


def install_procylingstats():
    """Install the procyclingstats library."""
    pip = get_pip_executable()
    
    # Check if procyclingstats exists in parent directory
    pcs_path = Path('../procyclingstats')
    
    if not pcs_path.exists():
        print("⚠️  procyclingstats not found in parent directory")
        print("   Expected: ../procyclingstats")
        print("   Please clone the procyclingstats repository first")
        return False
    
    print("Installing procyclingstats library...")
    result = subprocess.run(
        [pip, 'install', '-e', '../procyclingstats'],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"❌ Failed to install procyclingstats:")
        print(result.stderr)
        return False
    
    print("✓ procyclingstats installed")
    return True


def apply_database_schema():
    """Apply database schema extensions."""
    python = get_python_executable()
    
    print("Applying database schema...")
    
    script = """
import sqlite3
import os

# Create data directory
os.makedirs('data', exist_ok=True)

# Connect to database (will create if doesn't exist)
conn = sqlite3.connect('data/cycling.db')

# Check if schema already applied
cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='rider_frailty'")
if cursor.fetchone():
    print("Schema already applied")
else:
    # Apply schema
    with open('genqirue/data/schema_extensions.sql', 'r') as f:
        conn.executescript(f.read())
    print("✓ Betting schema applied")

conn.close()
"""
    
    result = subprocess.run(
        [python, '-c', script],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"❌ Failed to apply schema:")
        print(result.stderr)
        return False
    
    print(result.stdout)
    return True


def run_tests():
    """Run basic tests."""
    python = get_python_executable()
    
    print("Running tests...")
    
    # Test imports
    script = """
try:
    from genqirue.models import GruppettoFrailtyModel
    from genqirue.portfolio import RobustKellyOptimizer
    print("✓ Core imports successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    exit(1)
"""
    
    result = subprocess.run(
        [python, '-c', script],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        return False
    
    print(result.stdout)
    return True


def create_env_file():
    """Create sample .env file."""
    env_content = """# Environment Configuration
# Copy this to .env and fill in your values

# Database (optional - defaults to SQLite)
# DATABASE_URL=postgresql://user:password@localhost:5432/cycling

# Slack notifications (optional)
# SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...

# Email alerts (optional)
# EMAIL_HOST=smtp.gmail.com
# EMAIL_USER=your-email@gmail.com
# EMAIL_PASSWORD=app-password
"""
    
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✓ Created .env file (edit with your settings)")
    else:
        print(".env file already exists")
    
    return True


def create_directories():
    """Create necessary directories."""
    dirs = ['data', 'logs', 'backups', 'migrations']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("✓ Created directories")
    return True


def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("\n1. Activate virtual environment:")
    if os.name == 'nt':
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    
    print("\n2. Run quick demo:")
    print("   python quickstart.py")
    
    print("\n3. Scrape some data:")
    print("   python -m pipeline.runner")
    
    print("\n4. Run full analysis:")
    print("   python example_betting_workflow.py")
    
    print("\n5. Read the docs:")
    print("   - README.md          - Getting started guide")
    print("   - COMMANDS.md        - Complete CLI reference")
    print("   - docs/ENGINE.md     - Model documentation")
    print("   - docs/SCRAPE.md     - Scraping pipeline details")
    
    print("\n" + "="*60)


def main():
    """Main setup function."""
    print("="*60)
    print("CYCLING PREDICT - TEAM SETUP")
    print("="*60)
    print("\nThis script will set up your development environment.")
    print("It may take a few minutes...\n")
    
    steps = [
        ("Checking Python version", check_python),
        ("Creating directories", create_directories),
        ("Creating virtual environment", create_venv),
        ("Installing dependencies", install_dependencies),
        ("Installing procyclingstats", install_procylingstats),
        ("Applying database schema", apply_database_schema),
        ("Running tests", run_tests),
        ("Creating environment file", create_env_file),
    ]
    
    for i, (description, func) in enumerate(steps, 1):
        print_step(i, description)
        if not func():
            print(f"\n❌ Setup failed at step {i}")
            print("Please fix the error and run again.")
            sys.exit(1)
    
    print_next_steps()


if __name__ == '__main__':
    main()
