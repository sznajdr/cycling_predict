"""
Live Race Dashboard - Personal Use
==================================
Streamlit app for live race monitoring and betting alerts.

Usage:
    streamlit run live_race_dashboard.py

Features:
- Race picker from configured races
- Live PCS scraping (positions, results)
- Model predictions display
- Attack alerts (Strategy 12 simulation)
- Value bet highlighting
"""
import streamlit as st
import sqlite3
import requests
import time
import pandas as pd
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import threading
import json

# Page config
st.set_page_config(
    page_title="Cycling Predict - Live",
    page_icon="🚴‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .live-indicator {
        color: #ff0000;
        animation: blink 1s infinite;
    }
    @keyframes blink {
        0% { opacity: 1; }
        50% { opacity: 0.3; }
        100% { opacity: 1; }
    }
    .value-bet {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 10px;
        margin: 5px 0;
    }
    .attack-alert {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 10px;
        margin: 5px 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


def get_db_connection():
    """Get database connection."""
    return sqlite3.connect('data/cycling.db', check_same_thread=False)


def load_races():
    """Load available races from database."""
    conn = get_db_connection()
    query = """
    SELECT r.id, r.display_name, r.year, r.pcs_slug, r.startdate, r.enddate,
           COUNT(DISTINCT rs.id) as num_stages,
           COUNT(DISTINCT sl.rider_id) as num_riders
    FROM races r
    LEFT JOIN race_stages rs ON rs.race_id = r.id
    LEFT JOIN startlist_entries sl ON sl.race_id = r.id
    GROUP BY r.id
    ORDER BY r.startdate DESC
    LIMIT 20
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def load_stages(race_id):
    """Load stages for selected race."""
    conn = get_db_connection()
    query = """
    SELECT rs.id, rs.stage_number, rs.stage_date, rs.stage_type, 
           rs.distance_km, rs.pcs_stage_url
    FROM race_stages rs
    WHERE rs.race_id = ?
    ORDER BY rs.stage_number
    """
    df = pd.read_sql_query(query, conn, params=(race_id,))
    conn.close()
    return df


def get_startlist(race_id):
    """Get race startlist with details."""
    conn = get_db_connection()
    query = """
    SELECT DISTINCT 
        r.id as rider_id,
        r.name,
        r.nationality,
        r.sp_climber,
        r.sp_sprint,
        r.sp_hills,
        r.sp_gc,
        t.name as team_name
    FROM startlist_entries sl
    JOIN riders r ON sl.rider_id = r.id
    JOIN teams t ON sl.team_id = t.id
    WHERE sl.race_id = ?
    ORDER BY r.sp_hills DESC, r.sp_sprint DESC
    LIMIT 50
    """
    df = pd.read_sql_query(query, conn, params=(race_id,))
    conn.close()
    return df


def scrape_live_results(pcs_slug, year, stage_num):
    """Scrape live results from PCS."""
    try:
        url = f"https://www.procyclingstats.com/race/{pcs_slug}/{year}/stage-{stage_num}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return None, f"HTTP {response.status_code}"
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Check if live
        is_live = 'live' in response.text.lower() or 'Live' in soup.get_text()
        
        # Try to extract results table
        results = []
        tables = soup.find_all('table')
        
        for table in tables:
            rows = table.find_all('tr')
            for row in rows[1:]:  # Skip header
                cols = row.find_all('td')
                if len(cols) >= 3:
                    rank = cols[0].get_text(strip=True)
                    rider = cols[1].get_text(strip=True)
                    time = cols[2].get_text(strip=True) if len(cols) > 2 else ""
                    results.append({
                        'rank': rank,
                        'rider': rider,
                        'time': time
                    })
        
        return {
            'is_live': is_live,
            'url': url,
            'timestamp': datetime.now(),
            'results': results[:20]  # Top 20
        }, None
        
    except Exception as e:
        return None, str(e)


def get_model_predictions(race_id, stage_id):
    """Get saved model predictions."""
    conn = get_db_connection()
    query = """
    SELECT r.name as rider_name, so.win_prob as model_prob, so.win_prob_std, so.edge_bps, 
           so.expected_value
    FROM strategy_outputs so
    JOIN riders r ON so.rider_id = r.id
    WHERE so.strategy_name = 'stage_ranking'
      AND so.stage_id = ?
    ORDER BY so.win_prob DESC
    LIMIT 20
    """
    df = pd.read_sql_query(query, conn, params=(stage_id,))
    conn.close()
    return df


def simulate_attack_detection():
    """Simulate Strategy 12 attack detection."""
    import random
    riders = ["Pogacar", "Ayuso", "Vingegaard", "Evenepoel", "Roglic"]
    rider = random.choice(riders)
    power = random.randint(380, 450)
    z_score = round(random.uniform(2.0, 3.5), 1)
    
    return {
        'rider': rider,
        'power': power,
        'z_score': z_score,
        'timestamp': datetime.now(),
        'confidence': min(95, int(70 + z_score * 10))
    }


def main():
    """Main Streamlit app."""
    
    # Header
    st.markdown('<p class="main-header">🚴‍♂️ Cycling Predict - Live Dashboard</p>', 
                unsafe_allow_html=True)
    
    # Sidebar - Race Selection
    st.sidebar.header("🏁 Race Selection")
    
    races_df = load_races()
    if races_df.empty:
        st.error("No races found in database. Run the scraper first!")
        st.stop()
    
    # Format race options
    race_options = []
    for _, row in races_df.iterrows():
        start = row['startdate'] if row['startdate'] else 'Unknown'
        label = f"{row['display_name']} {row['year']} ({row['num_stages']} stages, {row['num_riders']} riders)"
        race_options.append((label, row['id'], row['pcs_slug'], row['year']))
    
    selected_label = st.sidebar.selectbox(
        "Select Race",
        options=[r[0] for r in race_options],
        index=0
    )
    
    # Get selected race details
    selected_idx = [r[0] for r in race_options].index(selected_label)
    race_id = race_options[selected_idx][1]
    pcs_slug = race_options[selected_idx][2]
    year = race_options[selected_idx][3]
    
    # Load stages
    stages_df = load_stages(race_id)
    if stages_df.empty:
        st.error("No stages found for this race!")
        st.stop()
    
    # Stage selection
    stage_options = []
    for _, row in stages_df.iterrows():
        label = f"Stage {row['stage_number']} - {row['stage_type']} ({row['distance_km']}km)"
        stage_options.append((label, row['id'], row['stage_number']))
    
    selected_stage_label = st.sidebar.selectbox(
        "Select Stage",
        options=[s[0] for s in stage_options]
    )
    
    selected_stage_idx = [s[0] for s in stage_options].index(selected_stage_label)
    stage_id = stage_options[selected_stage_idx][1]
    stage_num = stage_options[selected_stage_idx][2]
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("🔄 Auto Refresh (30s)", value=False)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Live Race", "💰 Value Bets", "👥 Startlist", "⚡ Alerts"
    ])
    
    # Tab 1: Live Race
    with tab1:
        st.header(f"Live: {selected_label} - Stage {stage_num}")
        
        # Live scrape button
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("🔄 Refresh Live Data"):
                st.session_state['live_data'] = None
        
        with col2:
            live_url = f"https://www.procyclingstats.com/race/{pcs_slug}/{year}/stage-{stage_num}"
            st.markdown(f"[Open PCS Live]({live_url})", unsafe_allow_html=True)
        
        # Scrape live data
        if 'live_data' not in st.session_state or st.session_state.get('refresh'):
            with st.spinner("Scraping live data from PCS..."):
                live_data, error = scrape_live_results(pcs_slug, year, stage_num)
                if error:
                    st.warning(f"Could not scrape live data: {error}")
                    live_data = None
                st.session_state['live_data'] = live_data
                st.session_state['refresh'] = False
        
        live_data = st.session_state.get('live_data')
        
        # Display live status
        if live_data and live_data.get('is_live'):
            st.markdown('<h3><span class="live-indicator">●</span> LIVE</h3>', 
                       unsafe_allow_html=True)
            st.caption(f"Last update: {live_data['timestamp'].strftime('%H:%M:%S')}")
        else:
            st.info("Race not live yet or no data available. The page will populate when the race starts.")
        
        # Live results table
        if live_data and live_data.get('results'):
            st.subheader("Live Results")
            results_df = pd.DataFrame(live_data['results'])
            st.dataframe(results_df, use_container_width=True, height=400)
        else:
            st.info("No results data available yet. Check back when the race is underway.")
        
        # Race metadata
        stage_info = stages_df[stages_df['id'] == stage_id].iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Distance", f"{stage_info['distance_km']}km")
        with col2:
            st.metric("Type", stage_info['stage_type'].upper())
        with col3:
            st.metric("Stage", f"{stage_num}/{len(stages_df)}")
        with col4:
            st.metric("Date", stage_info['stage_date'])
    
    # Tab 2: Value Bets
    with tab2:
        st.header("💰 Model Predictions & Value Bets")
        
        # Check if we have model predictions
        predictions_df = get_model_predictions(race_id, stage_id)
        
        if predictions_df.empty:
            st.warning("No model predictions found. Run: `python rank_stage.py paris-nice 2026 1 --run-models`")
            
            if st.button("🚀 Run Analysis Now"):
                st.info("Run this command in your terminal:")
                st.code(f"python rank_stage.py {pcs_slug} {year} {stage_num} --run-models", 
                       language='bash')
        else:
            st.success(f"Found predictions for {len(predictions_df)} riders")
            
            # Value bets (edge > 50 bps)
            value_bets = predictions_df[predictions_df['edge_bps'] > 50].head(10)
            
            if not value_bets.empty:
                st.subheader("🎯 Top Value Opportunities (Edge > 50bps)")
                
                for _, row in value_bets.iterrows():
                    with st.container():
                        st.markdown(f"""
                        <div class="value-bet">
                            <strong>{row['rider_name']}</strong><br>
                            Model: {row['model_prob']:.1%} | Edge: +{row['edge_bps']:.0f} bps
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No value bets found with edge > 50bps. Check back after odds update.")
            
            # Full predictions table
            st.subheader("All Model Predictions")
            
            # Format dataframe
            display_df = predictions_df.copy()
            display_df['model_prob'] = display_df['model_prob'].apply(lambda x: f"{x:.1%}")
            display_df['edge_bps'] = display_df['edge_bps'].apply(lambda x: f"{x:+.0f}")
            
            st.dataframe(display_df, use_container_width=True, height=500)
    
    # Tab 3: Startlist
    with tab3:
        st.header("👥 Race Startlist")
        
        startlist_df = get_startlist(race_id)
        
        if not startlist_df.empty:
            st.write(f"Top {len(startlist_df)} riders by specialty")
            
            # Format specialty scores
            for col in ['sp_climber', 'sp_sprint', 'sp_hills', 'sp_gc']:
                if col in startlist_df.columns:
                    startlist_df[col] = startlist_df[col].fillna(0).astype(int)
            
            st.dataframe(startlist_df, use_container_width=True, height=600)
            
            # Specialty distribution
            st.subheader("Specialty Distribution")
            col1, col2 = st.columns(2)
            
            with col1:
                st.bar_chart(startlist_df.set_index('name')['sp_hills'].head(15))
            
            with col2:
                st.bar_chart(startlist_df.set_index('name')['sp_sprint'].head(15))
        else:
            st.error("No startlist found!")
    
    # Tab 4: Alerts
    with tab4:
        st.header("⚡ Live Alerts")
        
        # Simulated attack alert
        if st.button("🚨 Simulate Attack Alert (Strategy 12)"):
            attack = simulate_attack_detection()
            st.session_state['last_attack'] = attack
        
        if 'last_attack' in st.session_state:
            attack = st.session_state['last_attack']
            st.markdown(f"""
            <div class="attack-alert">
                <h4>🚨 ATTACK DETECTED</h4>
                <strong>Rider:</strong> {attack['rider']}<br>
                <strong>Power:</strong> {attack['power']}W<br>
                <strong>Z-Score:</strong> {attack['z_score']}<br>
                <strong>Confidence:</strong> {attack['confidence']}%<br>
                <strong>Time:</strong> {attack['timestamp'].strftime('%H:%M:%S')}
            </div>
            """, unsafe_allow_html=True)
            
            st.info("💡 Strategy 12 (BOCPD) detected a structural power increase. Consider in-play bet.")
        
        # Alert log
        st.subheader("Alert Log")
        st.info("Attack detection alerts will appear here when the race is live.")
        
        # Manual refresh trigger
        if auto_refresh:
            time.sleep(30)
            st.session_state['refresh'] = True
            st.rerun()


if __name__ == "__main__":
    main()
