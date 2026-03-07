# Deployment Guide

How to run Cycling Predict in production — shared databases, cloud hosting, automated scheduling, monitoring, and backups.

---

## Table of Contents

1. [Shared Database](#1-shared-database)
2. [Cloud Deployment](#2-cloud-deployment)
3. [Automated Scraping](#3-automated-scraping)
4. [Monitoring and Alerts](#4-monitoring-and-alerts)
5. [Backups](#5-backups)
6. [Production Checklist](#6-production-checklist)

---

## 1. Shared Database

SQLite is the default and works well for a single user. For teams, switch to PostgreSQL.

### PostgreSQL setup

**Local:**

```bash
createdb cycling_predict
createuser -P cycling_user
```

**Cloud (Railway, Supabase, AWS RDS, DigitalOcean Managed DB):**

Sign up for any provider, create a PostgreSQL instance, and save the connection string.

**Configure via environment variable:**

Create `.env` (already in `.gitignore`):

```bash
DATABASE_URL=postgresql://cycling_user:your-password@localhost:5432/cycling_predict
```

Update `pipeline/db.py` to read the environment variable:

```python
import os
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///data/cycling.db')
```

Install the driver:

```bash
pip install asyncpg
```

### ClickHouse (high-frequency analytics)

Suitable if storing live telemetry at high volume. See the ClickHouse documentation for schema migration. Connection:

```bash
pip install clickhouse-driver
```

---

## 2. Cloud Deployment

### Option A: VPS (DigitalOcean, Linode, AWS EC2)

Recommended minimum: 2 GB RAM, 50 GB storage, Ubuntu 22.04.

```bash
ssh user@YOUR_SERVER_IP

apt update && apt install python3-pip python3-venv git -y

git clone https://github.com/your-org/cycling-predict.git
cd cycling-predict

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e ../procyclingstats
```

Run as a systemd service. Create `/etc/systemd/system/cycling-scraper.service`:

```ini
[Unit]
Description=Cycling Predict Scraper
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/cycling-predict
Environment=PYTHONPATH=/home/ubuntu/cycling-predict
ExecStart=/home/ubuntu/cycling-predict/venv/bin/python -m pipeline.runner
Restart=on-failure
RestartSec=3600

[Install]
WantedBy=multi-user.target
```

```bash
systemctl enable cycling-scraper
systemctl start cycling-scraper
systemctl status cycling-scraper
```

### Option B: Docker

`Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-m", "pipeline.runner"]
```

`docker-compose.yml`:

```yaml
version: '3.8'

services:
  scraper:
    build: .
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - DATABASE_URL=postgresql://postgres:your-password@db:5432/cycling
    depends_on:
      - db
    restart: unless-stopped

  db:
    image: postgres:15
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: your-password
      POSTGRES_DB: cycling
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

volumes:
  postgres_data:
```

```bash
docker-compose up -d
```

---

## 3. Automated Scraping

### Cron — PCS scraper (Linux/macOS)

```bash
crontab -e

# Daily at 06:00 and 18:00
0 6,18 * * * cd /path/to/cycling-predict && venv/bin/python -m pipeline.runner >> logs/cron.log 2>&1
```

### Cron — Betclic odds (Linux/macOS)

```bash
# Every 30 minutes, 06:00–22:00
*/30 6-22 * * * cd /path/to/cycling-predict && venv/bin/python fetch_odds.py >> logs/odds.log 2>&1
```

### Windows Task Scheduler

1. Open Task Scheduler → Create Basic Task
2. Trigger: Daily at 06:00, repeat every 30 minutes (for odds)
3. Action: Start a program — Program: `python`, Arguments: `-m pipeline.runner` (or `fetch_odds.py`), Start in: `path\to\cycling-predict`

### GitHub Actions

See `.github/workflows/scrape.yml` — triggered daily at 06:00 UTC and manually via workflow dispatch.

---

## 4. Monitoring and Alerts

### Permanent-fail check

```python
import sqlite3

conn = sqlite3.connect('data/cycling.db')
failed = conn.execute("SELECT COUNT(*) FROM fetch_queue WHERE status='permanent_fail'").fetchone()[0]
conn.close()

if failed > 0:
    print(f"WARNING: {failed} jobs in permanent_fail state")
```

Add to a cron job or CI step.

### Webhook notification (Slack, Discord, generic)

```python
import os, urllib.request, json

def notify(message: str) -> None:
    url = os.environ.get('WEBHOOK_URL')
    if not url:
        return
    data = json.dumps({'text': message}).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={'Content-Type': 'application/json'})
    urllib.request.urlopen(req)
```

Set `WEBHOOK_URL` in your `.env` file.

### Simple status endpoint

```python
# scripts/dashboard.py
from flask import Flask, jsonify
import sqlite3

app = Flask(__name__)

@app.route('/status')
def status():
    conn = sqlite3.connect('data/cycling.db')
    row = conn.execute("""
        SELECT
            (SELECT COUNT(*) FROM races)                                    AS races,
            (SELECT COUNT(*) FROM riders)                                   AS riders,
            (SELECT COUNT(*) FROM rider_results)                            AS results,
            (SELECT COUNT(*) FROM fetch_queue WHERE status='pending')       AS pending,
            (SELECT COUNT(*) FROM fetch_queue WHERE status='permanent_fail') AS failed
    """).fetchone()
    conn.close()
    return jsonify(dict(zip(['races','riders','results','pending','failed'], row)))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

```bash
pip install flask
python scripts/dashboard.py
# GET http://localhost:5000/status
```

---

## 5. Backups

### Daily local backup

```bash
#!/bin/bash
# scripts/backup.sh
DATE=$(date +%Y%m%d)
mkdir -p backups
cp data/cycling.db backups/cycling_$DATE.db
gzip backups/cycling_$DATE.db
find backups/ -name "*.gz" -mtime +7 -delete   # keep 7 days
```

Add to crontab: `0 2 * * * /path/to/cycling-predict/scripts/backup.sh`

### Cloud backup (AWS S3)

```bash
pip install awscli
aws configure

# Add to backup.sh
aws s3 cp backups/cycling_$DATE.db.gz s3://your-bucket/cycling-backups/
```

---

## 6. Production Checklist

- [ ] Switch to PostgreSQL (SQLite has write-concurrency limits)
- [ ] Store all secrets in environment variables, not code
- [ ] Enable logging to file (`logs/pipeline.log` is pre-configured)
- [ ] Set up monitoring (permanent_fail check or webhook)
- [ ] Configure daily backups
- [ ] Use Docker for reproducible deployment
- [ ] Enable branch protection on `main` (see `CONTRIBUTING.md`)
- [ ] Respect PCS rate limits — the scraper already enforces 1 req/s minimum

### Approximate costs

| Option | Monthly | Best for |
|--------|---------|---------|
| Local + GitHub free tier | Free | Development, single user |
| VPS 2 GB RAM | $5–12 | Always-on scraping, small team |
| AWS EC2 t3.small | ~$15 | AWS-native setup |
| Railway or Render | Free–$10 | Managed, easy deployment |
| Supabase PostgreSQL | Free tier | Managed shared database |
