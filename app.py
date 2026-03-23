"""
F1 Race Strategy Intelligence Dashboard — Phase 4
Run with:  streamlit run app.py
Requires:  pip install streamlit plotly pandas scikit-learn fastf1
Place this file in the same folder as your f1_data/ directory.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="F1 Strategy Intelligence",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f0f0f; }
    .stApp { background-color: #0f0f0f; color: #f0f0f0; }
    .metric-card {
        background: #1a1a2e;
        border: 1px solid #e10600;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #e10600; }
    .metric-label { font-size: 0.85rem; color: #aaaaaa; margin-top: 4px; }
    .section-header {
        font-size: 1.2rem; font-weight: 600;
        color: #e10600; border-bottom: 1px solid #333;
        padding-bottom: 6px; margin-bottom: 16px;
    }
    div[data-testid="stSidebar"] { background-color: #111111; }
    div[data-testid="stSidebar"] * { color: #f0f0f0 !important; }
    .stSelectbox label, .stSlider label, .stMultiSelect label { color: #cccccc !important; }
    .podium-high { color: #00d2be; font-weight: 700; }
    .podium-low  { color: #888888; }
</style>
""", unsafe_allow_html=True)

# ── Team colours ──────────────────────────────────────────────────────────────
TEAM_COLORS = {
    "Red Bull Racing": "#3671C6", "Mercedes": "#00D2BE", "Ferrari": "#E8002D",
    "McLaren": "#FF8000", "Aston Martin": "#358C75", "Alpine": "#2293D1",
    "Williams": "#37BEDD", "AlphaTauri": "#5E8FAA",
    "Alfa Romeo": "#C92D4B", "Haas F1 Team": "#B6BABD",
}
COMPOUND_COLORS = {"SOFT": "#e10600", "MEDIUM": "#ffd700", "HARD": "#eeeeee"}

# ── Data loading & caching ────────────────────────────────────────────────────
@st.cache_data
def load_data():
    DATA_DIR = "f1_data"
    laps    = pd.read_csv(f"{DATA_DIR}/2023_laps.csv")
    pits    = pd.read_csv(f"{DATA_DIR}/2023_pits.csv")
    results = pd.read_csv(f"{DATA_DIR}/2023_results.csv")
    weather = pd.read_csv(f"{DATA_DIR}/2023_weather.csv")

    # Clean laps
    laps_clean = laps.dropna(subset=["LapTimeSec"]).copy()
    median_lap  = laps_clean["LapTimeSec"].median()
    laps_clean  = laps_clean[
        (laps_clean["LapTimeSec"] < median_lap * 1.30) &
        (laps_clean["TrackStatus"] == 1)
    ]
    laps_racing = laps_clean[
        laps_clean["PitInTime"].isna() & laps_clean["PitOutTime"].isna()
    ].copy()

    # Clean weather
    weather["AirTemp"]   = pd.to_numeric(weather["AirTemp"],   errors="coerce")
    weather["TrackTemp"] = pd.to_numeric(weather["TrackTemp"], errors="coerce")
    weather["WindSpeed"] = pd.to_numeric(weather["WindSpeed"], errors="coerce")
    weather_avg = weather.groupby("EventName")[["AirTemp","TrackTemp","WindSpeed"]].mean().reset_index()

    # Build ML feature table
    stops = pits.groupby(["EventName","Driver"]).size().reset_index(name="NumStops")
    tyre_max = laps.groupby(["EventName","Driver"])["TyreLife"].max().reset_index(name="MaxTyreLife")
    constructor_pts = results.groupby("TeamName")["Points"].sum().reset_index(name="ConstructorPts")
    team_rank = constructor_pts.sort_values("ConstructorPts").reset_index(drop=True)
    team_rank["TeamRank"] = team_rank.index + 1

    df = results[["EventName","Abbreviation","TeamName","GridPosition","Position","Status"]].copy()
    df["Podium"] = (df["Position"] <= 3).astype(int)
    df["Outcome"] = df["Status"].map(
        lambda s: "Finished" if s == "Finished" else ("Lapped" if s == "Lapped" else "DNF")
    )
    df["PositionDelta"] = df["GridPosition"] - df["Position"]
    df = df.merge(stops,           left_on=["EventName","Abbreviation"], right_on=["EventName","Driver"], how="left")
    df = df.merge(weather_avg,     on="EventName",                                                         how="left")
    df = df.merge(tyre_max,        left_on=["EventName","Abbreviation"], right_on=["EventName","Driver"], how="left")
    df = df.merge(constructor_pts, on="TeamName",                                                          how="left")
    df = df.merge(team_rank[["TeamName","TeamRank"]], on="TeamName",                                       how="left")
    df["NumStops"]    = df["NumStops"].fillna(0)
    df["MaxTyreLife"] = df["MaxTyreLife"].fillna(df["MaxTyreLife"].median())
    df["GridTop5"]    = (df["GridPosition"] <= 5).astype(int)

    # Driver pace rating
    driver_pace = (
        laps_racing
        .groupby(["EventName","Driver","Team"])["LapTimeSec"]
        .median().reset_index(name="MedianLap")
    )
    team_best = driver_pace.groupby(["EventName","Team"])["MedianLap"].min().reset_index(name="TeamBestLap")
    driver_pace = driver_pace.merge(team_best, on=["EventName","Team"])
    driver_pace["GapToTeammate"] = driver_pace["MedianLap"] - driver_pace["TeamBestLap"]
    driver_rating = (
        driver_pace.groupby("Driver")
        .agg(AvgGap=("GapToTeammate","mean"), Team=("Team","first"), Races=("EventName","nunique"))
        .reset_index().sort_values("AvgGap")
    )

    return df, laps_racing, pits, results, weather, weather_avg, constructor_pts, team_rank, driver_rating

@st.cache_resource
def train_model(df):
    FEATURES = ["GridPosition","GridTop5","NumStops","TrackTemp","MaxTyreLife","TeamRank","ConstructorPts"]
    X = df[FEATURES].dropna()
    y = df.loc[X.index, "Podium"]
    model = LogisticRegression(max_iter=1000, C=0.5, class_weight="balanced", random_state=42)
    model.fit(X, y)
    return model, FEATURES

# ── Load everything ───────────────────────────────────────────────────────────
df, laps_racing, pits, results, weather, weather_avg, constructor_pts, team_rank, driver_rating = load_data()
model, FEATURES = train_model(df)
df["PodiumProb"] = model.predict_proba(df[FEATURES].fillna(df[FEATURES].median()))[:,1]

EVENTS  = sorted(df["EventName"].unique())
DRIVERS = sorted(df["Abbreviation"].unique())
TEAMS   = sorted(df["TeamName"].unique())

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏎️ F1 Intelligence")
    st.markdown("**2023 Season · Rounds 1–3**")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏁 Season Overview", "🔢 Driver vs Driver", "🛞 Tire Strategy",
         "📉 Lap Time Analysis", "🤖 Podium Predictor"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**Rounds covered**")
    for e in EVENTS:
        st.markdown(f"- {e}")
    st.markdown("---")
    st.caption("Built with FastF1 · scikit-learn · Streamlit")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — SEASON OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏁 Season Overview":
    st.markdown("## Season Overview")
    st.markdown("Top-level picture of the 2023 season across the first 3 rounds.")

    # KPI row
    finished     = results[results["Status"] == "Finished"]
    dnf_rate     = (results["Status"] == "Retired").mean() * 100
    avg_stops    = pits.groupby(["EventName","Driver"]).size().mean()
    top_team     = constructor_pts.sort_values("ConstructorPts", ascending=False).iloc[0]
    top_driver   = results.groupby("Abbreviation")["Points"].sum().idxmax()

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in [
        (c1, f"{top_team['TeamName']}", "Leading constructor"),
        (c2, f"{top_driver}", "Points leader (driver)"),
        (c3, f"{dnf_rate:.0f}%", "DNF rate"),
        (c4, f"{avg_stops:.1f}", "Avg pit stops / race"),
    ]:
        col.markdown(
            f'<div class="metric-card"><div class="metric-value">{val}</div>'
            f'<div class="metric-label">{label}</div></div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # Constructor points bar
    with col1:
        st.markdown('<div class="section-header">Constructor standings</div>', unsafe_allow_html=True)
        cp = constructor_pts.sort_values("ConstructorPts", ascending=True)
        fig = px.bar(
            cp, x="ConstructorPts", y="TeamName", orientation="h",
            color="TeamName",
            color_discrete_map=TEAM_COLORS,
            labels={"ConstructorPts": "Points", "TeamName": ""},
            template="plotly_dark",
        )
        fig.update_layout(
            showlegend=False, height=360,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=20, t=10, b=10),
            xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Outcome pie
    with col2:
        st.markdown('<div class="section-header">Race outcome breakdown</div>', unsafe_allow_html=True)
        outcome_counts = results["Status"].map(
            lambda s: "Finished" if s=="Finished" else ("Lapped" if s=="Lapped" else "DNF")
        ).value_counts().reset_index()
        outcome_counts.columns = ["Outcome","Count"]
        fig2 = px.pie(
            outcome_counts, names="Outcome", values="Count",
            color="Outcome",
            color_discrete_map={"Finished": "#2ecc71","Lapped": "#f39c12","DNF": "#e10600"},
            template="plotly_dark", hole=0.4,
        )
        fig2.update_layout(
            height=360, paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Position delta
    st.markdown('<div class="section-header">Positions gained / lost (grid → finish)</div>', unsafe_allow_html=True)
    delta = (
        df[df["Outcome"] == "Finished"]
        .groupby("Abbreviation")["PositionDelta"]
        .mean().reset_index().sort_values("PositionDelta", ascending=False)
    )
    delta["Color"] = delta["PositionDelta"].apply(lambda v: "#2ecc71" if v > 0 else "#e10600" if v < 0 else "#888")
    fig3 = px.bar(
        delta, x="Abbreviation", y="PositionDelta",
        color="Color", color_discrete_map={c:c for c in delta["Color"].unique()},
        labels={"PositionDelta": "Avg positions gained (+) / lost (−)", "Abbreviation": "Driver"},
        template="plotly_dark",
    )
    fig3.add_hline(y=0, line_color="gray", line_dash="dash", line_width=1)
    fig3.update_layout(
        showlegend=False, height=300,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333"),
        margin=dict(l=0, r=0, t=10, b=10),
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Driver rating
    st.markdown('<div class="section-header">Driver pace rating — gap to fastest teammate (lower = faster)</div>', unsafe_allow_html=True)
    dr = driver_rating.copy()
    dr["TeamColor"] = dr["Team"].map(lambda t: TEAM_COLORS.get(t, "#888888"))
    fig4 = px.bar(
        dr, x="AvgGap", y="Driver", orientation="h",
        color="Team", color_discrete_map=TEAM_COLORS,
        labels={"AvgGap": "Avg gap to fastest teammate (s)", "Driver": ""},
        template="plotly_dark",
    )
    fig4.update_layout(
        height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333"),
        margin=dict(l=0, r=0, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DRIVER VS DRIVER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔢 Driver vs Driver":
    st.markdown("## Driver Head-to-Head Comparison")

    col1, col2, col3 = st.columns([2, 2, 3])
    with col1:
        driver_a = st.selectbox("Driver A", DRIVERS, index=DRIVERS.index("VER"))
    with col2:
        driver_b = st.selectbox("Driver B", DRIVERS, index=DRIVERS.index("PER"))
    with col3:
        selected_events = st.multiselect("Races", EVENTS, default=EVENTS)

    if not selected_events:
        st.warning("Select at least one race.")
        st.stop()

    sub = df[df["EventName"].isin(selected_events)]
    ra  = sub[sub["Abbreviation"] == driver_a]
    rb  = sub[sub["Abbreviation"] == driver_b]

    # KPI row
    def driver_kpis(d):
        total_pts  = d["Points"].sum() if "Points" in d.columns else \
                     results[results["Abbreviation"]==d["Abbreviation"].iloc[0]]["Points"].sum()
        avg_grid   = d["GridPosition"].mean()
        avg_finish = d["Position"].mean()
        podiums    = d["Podium"].sum()
        return avg_grid, avg_finish, podiums

    ag_a, af_a, pod_a = driver_kpis(ra)
    ag_b, af_b, pod_b = driver_kpis(rb)

    pts_a = results[results["Abbreviation"]==driver_a]["Points"].sum()
    pts_b = results[results["Abbreviation"]==driver_b]["Points"].sum()

    st.markdown("<br>", unsafe_allow_html=True)
    cols = st.columns(5)
    metrics = [
        ("Points", f"{pts_a:.0f}", f"{pts_b:.0f}"),
        ("Avg grid pos", f"{ag_a:.1f}", f"{ag_b:.1f}"),
        ("Avg finish pos", f"{af_a:.1f}", f"{af_b:.1f}"),
        ("Podiums", f"{int(pod_a)}", f"{int(pod_b)}"),
    ]
    cols[0].markdown(f"**Metric**")
    cols[1].markdown(f"**{driver_a}**")
    cols[2].markdown("**vs**")
    cols[3].markdown(f"**{driver_b}**")
    for c, (label, va, vb) in zip([cols[0], cols[1], cols[3]], metrics):
        pass  # replaced by table below

    compare_df = pd.DataFrame(metrics, columns=["Metric", driver_a, driver_b])
    st.dataframe(compare_df.set_index("Metric"), use_container_width=False)

    col1, col2 = st.columns(2)

    # Grid vs Finish scatter
    with col1:
        st.markdown('<div class="section-header">Grid position vs finishing position</div>', unsafe_allow_html=True)
        fig = go.Figure()
        for driver, color in [(driver_a, "#e10600"), (driver_b, "#00D2BE")]:
            d = sub[sub["Abbreviation"] == driver]
            fig.add_trace(go.Scatter(
                x=d["GridPosition"], y=d["Position"],
                mode="markers+text",
                name=driver,
                text=d["EventName"].str.replace(" Grand Prix",""),
                textposition="top center",
                textfont=dict(size=9),
                marker=dict(size=14, color=color, line=dict(width=1, color="white")),
            ))
        fig.add_shape(type="line", x0=1, y0=1, x1=20, y1=20,
                      line=dict(color="gray", dash="dash", width=1))
        fig.update_layout(
            template="plotly_dark", height=320,
            xaxis_title="Grid position", yaxis_title="Finishing position",
            xaxis=dict(autorange="reversed", gridcolor="#333"),
            yaxis=dict(autorange="reversed", gridcolor="#333"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Lap time box per event
    with col2:
        st.markdown('<div class="section-header">Race pace distribution (racing laps)</div>', unsafe_allow_html=True)
        fig2 = go.Figure()
        for driver, color in [(driver_a, "#e10600"), (driver_b, "#00D2BE")]:
            d = laps_racing[laps_racing["Driver"] == driver]
            d = d[d["EventName"].isin(selected_events)]
            fig2.add_trace(go.Box(
                y=d["LapTimeSec"], name=driver,
                marker_color=color, boxmean=True,
                line=dict(width=1.5),
            ))
        fig2.update_layout(
            template="plotly_dark", height=320,
            yaxis_title="Lap time (s)",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(gridcolor="#333"),
            margin=dict(l=0, r=0, t=10, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Lap-by-lap evolution
    st.markdown('<div class="section-header">Lap-by-lap pace evolution</div>', unsafe_allow_html=True)
    event_choice = st.selectbox("Select race", selected_events)
    fig3 = go.Figure()
    for driver, color in [(driver_a, "#e10600"), (driver_b, "#00D2BE")]:
        d = laps_racing[(laps_racing["Driver"]==driver) & (laps_racing["EventName"]==event_choice)]
        d = d.sort_values("LapNumber")
        fig3.add_trace(go.Scatter(
            x=d["LapNumber"], y=d["LapTimeSec"],
            mode="lines+markers", name=driver,
            line=dict(color=color, width=2),
            marker=dict(size=4),
        ))
    fig3.update_layout(
        template="plotly_dark", height=300,
        xaxis_title="Lap number", yaxis_title="Lap time (s)",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333"),
        margin=dict(l=0, r=0, t=10, b=10),
    )
    st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — TIRE STRATEGY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🛞 Tire Strategy":
    st.markdown("## Tire Strategy Analysis")

    col1, col2 = st.columns(2)

    # Compound lap time violin
    with col1:
        st.markdown('<div class="section-header">Lap time by compound</div>', unsafe_allow_html=True)
        fig = go.Figure()
        for compound in ["SOFT","MEDIUM","HARD"]:
            d = laps_racing[laps_racing["Compound"]==compound]["LapTimeSec"].dropna()
            fig.add_trace(go.Violin(
                y=d, name=compound,
                fillcolor=COMPOUND_COLORS[compound],
                line_color="#555555",
                opacity=0.85, box_visible=True, meanline_visible=True,
            ))
        fig.update_layout(
            template="plotly_dark", height=360,
            yaxis_title="Lap time (s)",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(gridcolor="#333"),
            margin=dict(l=0,r=0,t=10,b=10), showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Compound usage per circuit
    with col2:
        st.markdown('<div class="section-header">Compound usage by circuit</div>', unsafe_allow_html=True)
        usage = (
            laps_racing.groupby(["EventName","Compound"]).size()
            .reset_index(name="Laps")
        )
        usage_pct = usage.copy()
        totals = usage.groupby("EventName")["Laps"].transform("sum")
        usage_pct["Pct"] = usage["Laps"] / totals * 100

        fig2 = px.bar(
            usage_pct, x="EventName", y="Pct", color="Compound",
            color_discrete_map=COMPOUND_COLORS,
            labels={"Pct":"% of racing laps","EventName":""},
            template="plotly_dark", barmode="stack",
            category_orders={"Compound":["SOFT","MEDIUM","HARD"]},
        )
        fig2.update_layout(
            height=360, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="#333",tickangle=15), yaxis=dict(gridcolor="#333"),
            margin=dict(l=0,r=0,t=10,b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Degradation curves
    st.markdown('<div class="section-header">Tire degradation — lap time vs tyre age</div>', unsafe_allow_html=True)
    event_filter = st.selectbox("Filter by circuit", ["All circuits"] + list(EVENTS))
    lr = laps_racing.copy()
    if event_filter != "All circuits":
        lr = lr[lr["EventName"] == event_filter]

    deg = lr.groupby(["Compound","TyreLife"])["LapTimeSec"].median().reset_index()
    counts = lr.groupby(["Compound","TyreLife"]).size().reset_index(name="n")
    deg = deg.merge(counts, on=["Compound","TyreLife"])
    deg = deg[deg["n"] >= 3]

    fig3 = go.Figure()
    for compound in ["SOFT","MEDIUM","HARD"]:
        d = deg[deg["Compound"]==compound].sort_values("TyreLife")
        if d.empty: continue
        fig3.add_trace(go.Scatter(
            x=d["TyreLife"], y=d["LapTimeSec"],
            mode="lines+markers", name=compound,
            line=dict(color=COMPOUND_COLORS[compound], width=2.5),
            marker=dict(size=5),
        ))
    fig3.update_layout(
        template="plotly_dark", height=320,
        xaxis_title="Tyre age (laps on compound)", yaxis_title="Median lap time (s)",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333"),
        margin=dict(l=0,r=0,t=10,b=10),
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Pit stop count per driver
    st.markdown('<div class="section-header">Pit stops per driver per race</div>', unsafe_allow_html=True)
    stops_per = pits.groupby(["EventName","Driver"]).size().reset_index(name="NumStops")
    stops_per = stops_per.merge(
        results[["EventName","Abbreviation","TeamName"]].rename(columns={"Abbreviation":"Driver"}),
        on=["EventName","Driver"], how="left"
    )
    fig4 = px.bar(
        stops_per, x="Driver", y="NumStops", color="EventName",
        barmode="group",
        labels={"NumStops":"Number of pit stops","Driver":""},
        template="plotly_dark",
    )
    fig4.update_layout(
        height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333"),
        margin=dict(l=0,r=0,t=10,b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — LAP TIME ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📉 Lap Time Analysis":
    st.markdown("## Lap Time Analysis")

    col1, col2 = st.columns([3, 2])
    with col1:
        selected_event = st.selectbox("Select race", EVENTS)
    with col2:
        selected_drivers = st.multiselect(
            "Highlight drivers", DRIVERS,
            default=["VER","PER","ALO"]
        )

    event_laps = laps_racing[laps_racing["EventName"] == selected_event].copy()

    # Lap time scatter coloured by compound
    st.markdown('<div class="section-header">Lap times coloured by tire compound</div>', unsafe_allow_html=True)
    fig = go.Figure()
    for compound in ["SOFT","MEDIUM","HARD"]:
        d = event_laps[event_laps["Compound"] == compound]
        fig.add_trace(go.Scatter(
            x=d["LapNumber"], y=d["LapTimeSec"],
            mode="markers", name=compound,
            marker=dict(
                color=COMPOUND_COLORS[compound],
                size=5, opacity=0.65,
                line=dict(width=0),
            ),
        ))
    # Overlay selected drivers
    for driver in selected_drivers:
        d = event_laps[event_laps["Driver"] == driver].sort_values("LapNumber")
        team = results[results["Abbreviation"]==driver]["TeamName"].iloc[0] if len(results[results["Abbreviation"]==driver]) > 0 else "Unknown"
        fig.add_trace(go.Scatter(
            x=d["LapNumber"], y=d["LapTimeSec"],
            mode="lines", name=f"{driver} (line)",
            line=dict(color=TEAM_COLORS.get(team,"#ffffff"), width=2),
        ))
    fig.update_layout(
        template="plotly_dark", height=360,
        xaxis_title="Lap number", yaxis_title="Lap time (s)",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333"),
        margin=dict(l=0,r=0,t=10,b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    # Sector time heatmap
    with col1:
        st.markdown('<div class="section-header">Avg sector times by driver</div>', unsafe_allow_html=True)
        sector_avg = (
            event_laps.groupby("Driver")[["Sector1TimeSec","Sector2TimeSec","Sector3TimeSec"]]
            .median().dropna()
        )
        sector_avg.columns = ["S1","S2","S3"]
        fig2 = px.imshow(
            sector_avg.T,
            color_continuous_scale="RdYlGn_r",
            labels={"color":"Median time (s)"},
            template="plotly_dark",
            aspect="auto",
        )
        fig2.update_layout(
            height=320, paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0,r=0,t=10,b=10),
            coloraxis_colorbar=dict(title="s"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Distribution by team
    with col2:
        st.markdown('<div class="section-header">Lap time distribution by team</div>', unsafe_allow_html=True)
        team_laps = event_laps.merge(
            results[["Abbreviation","TeamName"]].drop_duplicates(),
            left_on="Driver", right_on="Abbreviation", how="left"
        )
        team_median = team_laps.groupby("TeamName")["LapTimeSec"].median().sort_values()
        fig3 = px.box(
            team_laps.assign(TeamName=pd.Categorical(
                team_laps["TeamName"], categories=team_median.index, ordered=True
            )),
            x="TeamName", y="LapTimeSec",
            color="TeamName", color_discrete_map=TEAM_COLORS,
            template="plotly_dark",
            labels={"LapTimeSec":"Lap time (s)","TeamName":""},
        )
        fig3.update_layout(
            height=320, showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="#333", tickangle=30), yaxis=dict(gridcolor="#333"),
            margin=dict(l=0,r=0,t=10,b=10),
        )
        st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — PODIUM PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Podium Predictor":
    st.markdown("## Podium Predictor")
    st.markdown("Adjust the race parameters and see what the ML model predicts.")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown('<div class="section-header">Race parameters</div>', unsafe_allow_html=True)
        grid_pos      = st.slider("Qualifying / grid position", 1, 20, 3)
        num_stops     = st.slider("Number of pit stops", 1, 5, 2)
        track_temp    = st.slider("Track temperature (°C)", 20, 55, 31)
        max_tyre_life = st.slider("Longest stint (laps on one tyre)", 5, 55, 25)
        team_selected = st.selectbox("Constructor", TEAMS, index=TEAMS.index("Red Bull Racing"))

        team_pts  = constructor_pts[constructor_pts["TeamName"]==team_selected]["ConstructorPts"].values[0]
        team_rnk  = team_rank[team_rank["TeamName"]==team_selected]["TeamRank"].values[0]
        grid_top5 = int(grid_pos <= 5)

        scenario = pd.DataFrame([{
            "GridPosition":   grid_pos,
            "GridTop5":       grid_top5,
            "NumStops":       num_stops,
            "TrackTemp":      track_temp,
            "MaxTyreLife":    max_tyre_life,
            "TeamRank":       team_rnk,
            "ConstructorPts": team_pts,
        }])
        prob = model.predict_proba(scenario)[0, 1]
        pred = prob >= 0.5

    with col_right:
        st.markdown('<div class="section-header">Prediction</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=round(prob * 100, 1),
            number={"suffix": "%", "font": {"size": 48, "color": "#e10600"}},
            delta={"reference": 50, "increasing": {"color": "#2ecc71"}, "decreasing": {"color": "#e10600"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#555"},
                "bar":  {"color": "#e10600" if prob >= 0.5 else "#3498db", "thickness": 0.25},
                "bgcolor": "#1a1a2e",
                "borderwidth": 2,
                "bordercolor": "#333",
                "steps": [
                    {"range": [0,  33], "color": "#1a1a2e"},
                    {"range": [33, 66], "color": "#1e1e3a"},
                    {"range": [66,100], "color": "#231a1a"},
                ],
                "threshold": {
                    "line": {"color": "#ffffff", "width": 3},
                    "thickness": 0.8,
                    "value": 50,
                },
            },
            title={"text": "Podium probability", "font": {"color": "#aaa", "size": 14}},
        ))
        fig_gauge.update_layout(
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#f0f0f0"},
            margin=dict(l=20, r=20, t=30, b=10),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        verdict_color = "#2ecc71" if pred else "#e10600"
        verdict_text  = "🏆 PODIUM PREDICTED" if pred else "❌ NO PODIUM"
        st.markdown(
            f'<div style="text-align:center; font-size:1.5rem; font-weight:700; color:{verdict_color}; '
            f'padding:12px; border:2px solid {verdict_color}; border-radius:10px; margin-top:8px;">'
            f'{verdict_text}</div>',
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"**Team:** {team_selected} ({team_pts:.0f} pts, rank {team_rnk}/10)")
        st.markdown(f"**Grid position:** P{grid_pos} {'(top 5 ✓)' if grid_top5 else ''}")
        st.markdown(f"**Strategy:** {num_stops}-stop, longest stint {max_tyre_life} laps")
        st.markdown(f"**Track temp:** {track_temp}°C")

    # Sensitivity: probability across all grid positions for this setup
    st.markdown("---")
    st.markdown('<div class="section-header">How grid position affects probability (all else equal)</div>', unsafe_allow_html=True)
    grid_range = list(range(1, 21))
    probs_all = []
    for gp in grid_range:
        sc = pd.DataFrame([{
            "GridPosition": gp, "GridTop5": int(gp<=5),
            "NumStops": num_stops, "TrackTemp": track_temp,
            "MaxTyreLife": max_tyre_life, "TeamRank": team_rnk,
            "ConstructorPts": team_pts,
        }])
        probs_all.append(model.predict_proba(sc)[0, 1])

    fig_sens = go.Figure()
    fig_sens.add_trace(go.Scatter(
        x=grid_range, y=[p*100 for p in probs_all],
        mode="lines+markers",
        line=dict(color="#e10600", width=2.5),
        marker=dict(size=6, color="#e10600"),
        name="Podium probability",
    ))
    fig_sens.add_hline(y=50, line_color="white", line_dash="dash",
                       line_width=1, annotation_text="Decision boundary (50%)")
    fig_sens.add_vline(x=grid_pos, line_color="#f39c12", line_dash="dot",
                       line_width=2, annotation_text=f"Current: P{grid_pos}")
    fig_sens.update_layout(
        template="plotly_dark", height=300,
        xaxis_title="Grid position", yaxis_title="Podium probability (%)",
        xaxis=dict(tickvals=list(range(1,21)), gridcolor="#333"),
        yaxis=dict(range=[0, 105], gridcolor="#333"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0,r=0,t=10,b=10), showlegend=False,
    )
    try:
    st.plotly_chart(
        fig_sens,
        use_container_width=True,
        config={"responsive": True, "displayModeBar": False}
        )
    except Exception:
    st.warning("⚠️ This chart is not supported on your device")

    # Historical predictions table
    st.markdown('<div class="section-header">Model predictions on historical data</div>', unsafe_allow_html=True)
    show_df = df[["Abbreviation","TeamName","EventName","GridPosition",
                  "NumStops","Position","Podium","PodiumProb"]].copy()
    show_df["PodiumProb"] = show_df["PodiumProb"].round(3)
    show_df["Predicted"]  = (show_df["PodiumProb"] >= 0.5).astype(int)
    show_df["Correct"]    = show_df["Predicted"] == show_df["Podium"]
    show_df = show_df.sort_values("PodiumProb", ascending=False)
    show_df.columns = ["Driver","Team","Race","Grid","Stops","Finish","Actual Podium","Prob","Predicted","Correct"]
    st.dataframe(
        show_df.style
        .background_gradient(subset=["Prob"], cmap="RdYlGn")
        .applymap(lambda v: "color: #2ecc71" if v is True else "color: #e10600" if v is False else "", subset=["Correct"]),
        use_container_width=True, height=400,
    )
