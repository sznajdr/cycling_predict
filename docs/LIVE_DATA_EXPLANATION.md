# Why Live Scraping Doesn't Work (And What To Do)

## The Problem

**You can scrape historical races, but not live ones.**

### Why?

| Aspect | Historical Races | Live Races |
|--------|-----------------|------------|
| **Page type** | Static HTML | Dynamic, API-driven |
| **Protection** | None | Cloudflare + rate limiting |
| **Business model** | Free | Premium (PCS Pro) |
| **Bot detection** | None | Aggressive |

ProCyclingStats **aggressively protects live data** because:
1. It's their most valuable content
2. They sell premium live timing (PCS Pro subscription)
3. Live pages get hammered by bots during races
4. They have legal/licensing obligations to race organizers

---

## Technical Details

### What Happens When You Try Live Scraping

```
Your Request → Cloudflare → Blocks with 403
                ↓
          JavaScript Challenge
                ↓
          Rate Limiting
                ↓
          IP Ban (if aggressive)
```

### Why Historical Works

```
Your Request → Race Results Page (2024) → Returns HTML
                ↓
          Static, cached, no protection
```

---

## Workarounds (Ranked by Legitimacy)

### 1. Use Official Sources (Recommended ✅)

**PCS Pro** (€5-10/month)
- Official API access
- Live timing data
- Legal and reliable
- What bookmakers use

**Official Race Websites**
- Paris-Nice: https://www.paris-nice.fr/en/live
- Tour de France: https://www.letour.fr/en/live
- Often have better data than PCS

**Broadcasters**
- Eurosport/GCN apps have live timing
- Sometimes have APIs
- NOS, Sporza, RAI (national broadcasters)

### 2. Manual + Database Hybrid (What You Should Do)

**Your Current Setup:**
```
┌─────────────────────────────────────────────┐
│  Pre-Race (Works!)                          │
│  - Scrape startlists ✅                     │
│  - Run models (Strategy 1, 2) ✅            │
│  - Get odds ✅                              │
│  - Generate predictions ✅                  │
└─────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────┐
│  Live Race (Manual)                         │
│  - Watch PCS in browser                     │
│  - Use your pre-computed model picks        │
│  - Place bets based on live visuals         │
└─────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────┐
│  Post-Race (Works!)                         │
│  - Scrape results ✅                        │
│  - Update models ✅                         │
│  - Evaluate predictions ✅                  │
└─────────────────────────────────────────────┘
```

### 3. Browser Automation (Grey Area ⚠️)

Use Selenium/Playwright to control a real browser:

```python
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

options = Options()
options.add_argument("--headless")
driver = webdriver.Chrome(options=options)
driver.get("https://www.procyclingstats.com/race/paris-nice/2026/stage-1")
# ... extract data ...
```

**Pros:** Bypasses simple bot detection  
**Cons:** 
- Violates Terms of Service
- Slow (browser overhead)
- Can still be blocked
- Legally risky

### 4. Third-Party APIs (If Available)

| Service | Cost | Cycling Coverage |
|---------|------|------------------|
| PCS Pro | € | Excellent |
| Flashscore API | €€ | Good |
| Sportradar | €€€ | Excellent |
| NTT Data (official) | €€€€ | Race-specific |

---

## Recommended Workflow for Today

Since you can't scrape live, do this:

### Pre-Race (Now)
```powershell
# 1. Get your predictions ready
python rank_stage.py paris-nice 2026 1 --run-models --save

# 2. Get current odds
python fetch_odds.py

# 3. View your value bets
python scripts/race_viewer.py
```

### Live Race (Manual)
1. Open https://www.procyclingstats.com/race/paris-nice/2026/stage-1
2. Watch for:
   - Breakaway formation (first 20km)
   - Crash announcements
   - Weather changes
   - Attack on Côte de Chanteloup-les-Vignes
3. Reference your model picks from database
4. Place in-play bets if you see value

### Post-Race (Tonight)
```powershell
# Scrape results for model training
python -m pipeline.runner
```

---

## What You CAN Scrape (Your System Works!)

✅ **Startlists** - Before race
✅ **Historical results** - Any time
✅ **Rider profiles** - Any time
✅ **Stage metadata** - Before race
✅ **Betclic odds** - Works with fetch_odds.py
✅ **Race calendars** - Any time

❌ **Live positions** - Blocked
❌ **Live timing** - Blocked
❌ **Real-time results** - Blocked during race

---

## Long-Term Solutions

### Option A: PCS Pro Subscription
- Cost: ~€50/year
- Gets you official API
- Legal and reliable
- Worth it if serious about betting

### Option B: Focus on Pre-Race
- Most edge is in pre-race anyway
- Live betting is risky, fast markets
- Your models (Strategy 1, 2) are pre-race focused
- Bet before race starts based on model edge

### Option C: Build Live Detection (Limited)
Use indirect signals:
- Twitter/X race hashtags
- PCS social media posts
- Betting market movements
- TV audio transcription

---

## Summary

| Question | Answer |
|----------|--------|
| Why doesn't live work? | Cloudflare protection on live pages |
| Why does historical work? | Static pages, no protection |
| Can I fix it? | Not easily - it's by design |
| What should I do? | Use pre-race models + manual live viewing |
| Is PCS Pro worth it? | Yes, if you bet regularly |

---

## Bottom Line

**Your system is working correctly.** The limitation is by design from PCS, not a bug in your code.

**For today:**
1. Run `python scripts/race_viewer.py` to see your predictions
2. Open PCS in browser manually
3. Use your model + odds to find value
4. Place bets based on pre-race analysis

The pre-race edge is often bigger than live anyway! 🎯
