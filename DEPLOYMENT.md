# Deployment Guide

How to deploy Cycling Predict for production use and team collaboration.

## Table of Contents

1. [GitHub Setup](#github-setup)
2. [Shared Database (Recommended)](#shared-database-recommended)
3. [Cloud Deployment](#cloud-deployment)
4. [Automated Scraping](#automated-scraping)
5. [Monitoring & Alerts](#monitoring--alerts)

---

## GitHub Setup

### Step 1: Create Repository

1. Go to https://github.com/new
2. Name: `cycling-predict` (or your preferred name)
3. Make it **Private** (if sharing betting strategies)
4. Don't initialize with README (we already have one)

### Step 2: Push Local Code

```bash
# In your local cycling_predict folder
git remote add origin https://github.com/YOUR_USERNAME/cycling-predict.git
git branch -M main
git push -u origin main
```

### Step 3: Add Collaborators

1. Go to GitHub repo → Settings → Manage access
2. Click "Add people"
3. Invite your team member by username or email

### Step 4: Branch Protection (Recommended)

1. Settings → Branches
2. Add rule for `main` branch:
   - Require pull request reviews
   - Require status checks to pass
   - Restrict pushes to main

---

## Shared Database (Recommended)

SQLite is fine for single-user, but for teams use PostgreSQL or ClickHouse.

### Option A: PostgreSQL (Recommended)

#### 1. Create Database

**Local:**
```bash
# Install PostgreSQL
# Create database
createdb cycling_predict

# Create user
createuser -P cycling_user
```

**Cloud (Railway, Supabase, AWS RDS):**
- Sign up for free tier
- Create PostgreSQL instance
- Save connection string

#### 2. Update Connection

Create `.env` file:
```bash
# .env (add to .gitignore!)
DATABASE_URL=postgresql://cycling_user:password@localhost:5432/cycling_predict
# OR for cloud:
# DATABASE_URL=postgresql://user:pass@host.railway.app:5432/railway
```

#### 3. Install Driver

```bash
pip install asyncpg
```

#### 4. Update pipeline/db.py

```python
import os
from sqlalchemy import create_engine

def get_connection():
    database_url = os.getenv('DATABASE_URL', 'sqlite:///data/cycling.db')
    if database_url.startswith('postgresql'):
        engine = create_engine(database_url)
        return engine.connect()
    else:
        # SQLite fallback
        import sqlite3
        return sqlite3.connect('data/cycling.db')
```

### Option B: ClickHouse (For Large Scale)

Best for high-frequency data and analytics.

```bash
pip install clickhouse-driver
```

See: https://clickhouse.com/

---

## Cloud Deployment

### Option A: Run on VPS (DigitalOcean, Linode, AWS EC2)

**1. Create Server:**
- Ubuntu 22.04 LTS
- 2GB RAM minimum (4GB recommended)
- 50GB storage

**2. Setup:**
```bash
# SSH into server
ssh root@YOUR_SERVER_IP

# Install Python, Git
apt update
apt install python3-pip python3-venv git -y

# Clone repo
git clone https://github.com/YOUR_USERNAME/cycling-predict.git
cd cycling-predict

# Setup virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install procyclingstats (upload to server or install from local)
```

**3. Run Scraper as Service:**

Create `/etc/systemd/system/cycling-scraper.service`:
```ini
[Unit]
Description=Cycling Predict Scraper
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/cycling-predict
Environment=PYTHONPATH=/root/cycling-predict
ExecStart=/root/cycling-predict/venv/bin/python -m pipeline.runner
Restart=on-failure
RestartSec=3600

[Install]
WantedBy=multi-user.target
```

Enable:
```bash
systemctl enable cycling-scraper
systemctl start cycling-scraper
systemctl status cycling-scraper
```

### Option B: Docker (Recommended for Teams)

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Run scraper
CMD ["python", "-m", "pipeline.runner"]
```

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  scraper:
    build: .
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/cycling
    depends_on:
      - db
    restart: unless-stopped

  db:
    image: postgres:15
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
      POSTGRES_DB: cycling
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  # Optional: Web dashboard
  dashboard:
    build: .
    command: python -m http.server 8080
    ports:
      - "8080:8080"
    volumes:
      - ./data:/app/data

volumes:
  postgres_data:
```

Run:
```bash
docker-compose up -d
```

---

## Automated Scraping

### Schedule Regular Scraping

**Using Cron (Linux/Mac):**

```bash
# Edit crontab
crontab -e

# Add lines (runs daily at 6 AM and 6 PM)
0 6,18 * * * cd /path/to/cycling-predict && /path/to/venv/bin/python -m pipeline.runner >> logs/cron.log 2>&1
```

**Using Windows Task Scheduler:**

1. Open Task Scheduler
2. Create Basic Task
3. Trigger: Daily at 6:00 AM
4. Action: Start a program
5. Program: `python`
6. Arguments: `-m pipeline.runner`
7. Start in: `C:\path\to\cycling-predict`

**Using GitHub Actions (Free):**

Create `.github/workflows/scrape.yml`:

```yaml
name: Daily Data Scrape

on:
  schedule:
    - cron: '0 6 * * *'  # 6 AM UTC daily
  workflow_dispatch:  # Allow manual trigger

jobs:
  scrape:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -e ../procyclingstats
    
    - name: Run scraper
      run: python -m pipeline.runner
    
    - name: Upload database artifact
      uses: actions/upload-artifact@v3
      with:
        name: cycling-db
        path: data/cycling.db
```

---

## Monitoring & Alerts

### Option A: Simple Email Alerts

Create `scripts/alert.py`:

```python
import smtplib
from email.mime.text import MIMEText
import sqlite3

def check_for_errors():
    conn = sqlite3.connect('data/cycling.db')
    cursor = conn.execute(
        "SELECT COUNT(*) FROM fetch_queue WHERE status = 'permanent_fail'"
    )
    failed = cursor.fetchone()[0]
    conn.close()
    
    if failed > 0:
        send_alert(f"{failed} scraping jobs failed permanently")

def send_alert(message):
    msg = MIMEText(message)
    msg['Subject'] = 'Cycling Predict Alert'
    msg['From'] = 'alerts@yourdomain.com'
    msg['To'] = 'team@yourdomain.com'
    
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login('your-email@gmail.com', 'app-password')
    server.send_message(msg)
    server.quit()

if __name__ == '__main__':
    check_for_errors()
```

### Option B: Slack/Discord Notifications

```python
import requests
import os

def notify_slack(message):
    webhook_url = os.getenv('SLACK_WEBHOOK_URL')
    requests.post(webhook_url, json={'text': message})

# Call when scraper completes or fails
notify_slack("✅ Daily scrape complete: 45 races updated")
```

### Option C: Web Dashboard

Simple monitoring page:

```python
# dashboard.py
from flask import Flask, jsonify
import sqlite3

app = Flask(__name__)

@app.route('/api/status')
def status():
    conn = sqlite3.connect('data/cycling.db')
    cursor = conn.execute('''
        SELECT 
            (SELECT COUNT(*) FROM races) as races,
            (SELECT COUNT(*) FROM riders) as riders,
            (SELECT COUNT(*) FROM rider_results) as results,
            (SELECT COUNT(*) FROM fetch_queue WHERE status = 'pending') as pending,
            (SELECT COUNT(*) FROM fetch_queue WHERE status = 'permanent_fail') as failed
    ''')
    row = cursor.fetchone()
    conn.close()
    
    return jsonify({
        'races': row[0],
        'riders': row[1],
        'results': row[2],
        'jobs_pending': row[3],
        'jobs_failed': row[4]
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

Run: `python dashboard.py`

Visit: `http://localhost:5000/api/status`

---

## Backup Strategy

### Database Backup

**Automatic Daily Backup:**

```bash
#!/bin/bash
# backup.sh
DATE=$(date +%Y%m%d)
cp data/cycling.db backups/cycling_$DATE.db
gzip backups/cycling_$DATE.db
# Keep only last 7 days
find backups/ -name "*.gz" -mtime +7 -delete
```

Add to crontab:
```bash
0 2 * * * /path/to/backup.sh
```

### Cloud Backup (AWS S3)

```bash
pip install awscli
aws configure

# Add to backup.sh
aws s3 cp backups/cycling_$DATE.db.gz s3://your-bucket/backups/
```

---

## Production Checklist

Before going live:

- [ ] Use PostgreSQL instead of SQLite
- [ ] Set up environment variables (not hardcoded secrets)
- [ ] Enable logging to file
- [ ] Set up monitoring/alerts
- [ ] Configure backups
- [ ] Use Docker for consistent deployment
- [ ] Set up CI/CD (GitHub Actions)
- [ ] Document API endpoints
- [ ] Rate limiting on scraper (be nice to PCS)
- [ ] Error handling for network failures

---

## Cost Estimates

| Option | Monthly Cost | Best For |
|--------|-------------|----------|
| Local + GitHub | Free | Development, single user |
| DigitalOcean Droplet ($6-12) | $6-12 | Small team, 24/7 scraping |
| AWS EC2 (t3.small) | ~$15 | Scalable, integration with AWS |
| Railway/Render | Free-$10 | Easy deployment, auto-scaling |
| Supabase (Postgres) | Free tier | Managed database |

---

## Security Notes

1. **Never commit:**
   - `.env` files with passwords
   - `data/cycling.db` (already in .gitignore)
   - API keys

2. **Use environment variables:**
   ```python
   import os
   db_password = os.getenv('DB_PASSWORD')
   ```

3. **Keep repository private** if it contains betting edge logic

---

## Questions?

- Infrastructure: Open an issue on GitHub
- Scraping issues: Check `SCRAPE_README.md`
- Model questions: Check `genqirue/README.md`
