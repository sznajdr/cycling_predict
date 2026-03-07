# GitHub Setup Guide

Complete guide to uploading this project to GitHub and collaborating with your team.

---

## Step 1: Create GitHub Repository

### Option A: Using GitHub Website (Recommended)

1. Go to https://github.com/new
2. **Repository name:** `cycling-predict` (or your preferred name)
3. **Description:** "Bayesian cycling betting engine with data scraping pipeline"
4. **Visibility:** Choose **Private** (recommended for betting strategies)
5. **DO NOT** check "Add a README file" (we already have one)
6. **DO NOT** check "Add .gitignore" (we already have one)
7. **DO NOT** check "Choose a license" (add later if needed)
8. Click **"Create repository"**

You'll see instructions. Copy the "push an existing repository" URL.

### Option B: Using GitHub CLI

```bash
# Install GitHub CLI first: https://cli.github.com/

# Login
ght auth login

# Create repo (private)
gh repo create cycling-predict --private --source=. --push
```

---

## Step 2: Push Your Local Code

Run these commands in your `cycling_predict` folder:

```powershell
# Add the GitHub repository as remote
git remote add origin https://github.com/YOUR_USERNAME/cycling-predict.git

# Rename branch to 'main' (GitHub standard)
git branch -M main

# Push code to GitHub
git push -u origin main
```

**Replace `YOUR_USERNAME` with your actual GitHub username.**

Example:
```powershell
git remote add origin https://github.com/your-username/cycling-predict.git
```

---

## Step 3: Verify Upload

1. Go to `https://github.com/YOUR_USERNAME/cycling-predict`
2. You should see all your files:
   - `README.md`
   - `genqirue/`
   - `pipeline/`
   - `config/`
   - etc.

---

## Step 4: Add Collaborator (Your Team Member)

1. On GitHub, go to your repository
2. Click **Settings** tab
3. Click **Collaborators** in left sidebar
4. Click **Add people**
5. Enter your teammate's GitHub username or email
6. Click **Add collaborator**
7. They'll receive an email invitation

### What Your Teammate Needs to Do

1. Accept the invitation (via email or GitHub notification)
2. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/cycling-predict.git
   cd cycling-predict
   ```
3. Run setup:
   ```bash
   python scripts/setup_team.py
   ```

---

## Step 5: Set Up Branch Protection (Recommended)

Prevent direct pushes to main:

1. Repository → Settings → Branches
2. Click **Add rule**
3. **Branch name pattern:** `main`
4. Check these options:
   - ☑️ **Require a pull request before merging**
   - ☑️ **Require approvals** (set to 1)
   - ☑️ **Require status checks to pass**
   - ☑️ **Require conversation resolution before merging**
5. Click **Create**

Now all changes must go through Pull Requests.

---

## Step 6: Create Issues/Milestones (Optional)

Track what you're working on:

1. Go to **Issues** tab
2. Click **New issue**
3. Create issues like:
   - "Implement Strategy 3: Medical PK model"
   - "Add Tour de France 2023 data"
   - "Fix frailty calculation bug"

---

## Daily Workflow

### Making Changes

```powershell
# 1. Pull latest changes
git pull origin main

# 2. Create new branch
git checkout -b feature/strategy-3-medical

# 3. Make your changes
# ... edit files ...

# 4. Stage changes
git add .

# 5. Commit
git commit -m "Implement Strategy 3: Medical PK model

- Two-compartment pharmacokinetic model
- Robust Kelly sizing
- Tests included"

# 6. Push branch
git push origin feature/strategy-3-medical

# 7. Create Pull Request on GitHub
```

### Creating a Pull Request

1. Go to GitHub repository
2. Click **"Compare & pull request"** (appears after push)
3. Add title and description
4. Request review from teammate
5. Click **Create pull request**
6. After approval, click **Merge**

---

## Sharing Data (Not in Git)

The database (`data/cycling.db`) is **not** in Git (too large, regenerated).

### Option 1: Export/Import Specific Races

```powershell
# Export a race
python scripts/export_race_data.py --race tour-de-france --year 2024

# This creates: tour-de-france_2024.zip
# Share this file via Google Drive, Dropbox, etc.

# Teammate imports:
python scripts/export_race_data.py --import-zip tour-de-france_2024.zip
```

### Option 2: Shared Cloud Database

Set up PostgreSQL on Railway or AWS (see DEPLOYMENT.md)

### Option 3: Sync via Cloud Storage

1. Upload `data/cycling.db` to Google Drive
2. Share link with teammate
3. Download and replace local database

---

## Troubleshooting

### "fatal: not a git repository"

```powershell
cd path/to/cycling_predict
git init
git add .
git commit -m "Initial commit"
```

### "remote origin already exists"

```powershell
# Remove old remote
git remote remove origin

# Add new one
git remote add origin https://github.com/YOUR_USERNAME/cycling-predict.git
```

### "failed to push some refs"

```powershell
# Pull first, then push
git pull origin main --rebase
git push origin main
```

### Authentication Issues

Use Personal Access Token:

1. GitHub → Settings → Developer settings → Personal access tokens
2. Generate new token (classic)
3. Select scopes: `repo`
4. Copy token
5. Use token as password when pushing:
   ```powershell
   git push origin main
   # Username: your-github-username
   # Password: your-personal-access-token
   ```

Or use GitHub CLI:
```powershell
gh auth login
```

---

## File Structure on GitHub

```
cycling-predict/           ← Repository root
├── .github/               ← GitHub Actions, issue templates
│   └── workflows/
│       └── scrape.yml     ← Automated scraping
├── .gitignore             ← Files excluded from Git
├── config/
│   └── races.yaml         ← Configure races to scrape
├── genqirue/              ← Betting models (main code)
├── pipeline/              ← Scraping code
├── scripts/               ← Utility scripts
├── tests/                 ← Test files
├── CONTRIBUTING.md        ← Team workflow guide
├── DEPLOYMENT.md          ← Production deployment
├── GITHUB_SETUP.md        ← This file
├── PLAN.md                ← Original 15 strategies spec
├── README.md              ← Main documentation
├── SCRAPE_README.md       ← Scraping details
├── example_betting_workflow.py
├── quickstart.py
└── requirements.txt
```

---

## Next Steps

After GitHub setup:

1. **Read CONTRIBUTING.md** - Team workflow guidelines
2. **Read DEPLOYMENT.md** - If deploying to production
3. **Run quickstart.py** - Verify everything works
4. **Start scraping** - `python -m pipeline.runner`

---

## Questions?

- Git issues: Check GitHub Docs (https://docs.github.com/)
- Project issues: Open an issue in your repository
- Setup issues: Run `python scripts/setup_team.py`
