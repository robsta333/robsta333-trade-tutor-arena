import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random

# ------------------------------
# Helpers: indicators & scenarios
# ------------------------------

def compute_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def compute_sma(series, window):
    return series.rolling(window=window).mean()

def generate_random_walk(n=60, start_price=100, drift=0.0, vol=1.0):
    prices = [start_price]
    for _ in range(n - 1):
        step = np.random.normal(drift, vol)
        prices.append(max(1, prices[-1] + step))
    return np.array(prices)

def ohlc_from_close(close_prices, noise=0.5):
    # Simple synthetic OHLC around close prices
    opens = close_prices + np.random.normal(0, noise, size=len(close_prices))
    highs = np.maximum(opens, close_prices) + np.abs(np.random.normal(0, noise, size=len(close_prices)))
    lows = np.minimum(opens, close_prices) - np.abs(np.random.normal(0, noise, size=len(close_prices)))
    closes = close_prices + np.random.normal(0, noise, size=len(close_prices))
    return opens, highs, lows, closes

def make_scenario(mode="Stocks", difficulty="Beginner"):
    """
    Generate a scenario:
    - mode: "Stocks" or "Crypto"
    - difficulty: "Beginner" / "Intermediate" / "Pro"
    Returns dict with:
      df (DataFrame), correct_direction, correct_trade, explanation, title
    """
    n_candles = 60
    # Difficulty adjusts trend strength
    if difficulty == "Beginner":
        trend_strength = 0.25
    elif difficulty == "Intermediate":
        trend_strength = 0.12
    else:  # Pro
        trend_strength = 0.06

    # Choose scenario type
    scenario_type = random.choice(["uptrend_pullback", "downtrend_bounce", "range_chop"])

    if scenario_type == "uptrend_pullback":
        drift = trend_strength
        correct_direction = "Up"
        correct_trade = "Go Long"
        base_title = "Uptrend with Pullback"
        explanation = (
            "Price is in a clear uptrend (higher lows, EMAs stacked bullish). "
            "The recent pullback towards the fast EMAs often favors a continuation long "
            "with stops under the pullback low rather than shorting into strength."
        )
    elif scenario_type == "downtrend_bounce":
        drift = -trend_strength
        correct_direction = "Down"
        correct_trade = "Go Short"
        base_title = "Downtrend with Weak Bounce"
        explanation = (
            "Price is in a clear downtrend (lower highs, EMAs stacked bearish). "
            "The latest bounce into the moving averages is more likely to fail, "
            "favoring a short into resistance with stops above the bounce high."
        )
    else:  # range_chop
        drift = 0.0
        correct_direction = "Choppy / Sideways"
        correct_trade = "Stay Flat"
        base_title = "Choppy Range"
        explanation = (
            "Price is ranging with overlapping candles and flat moving averages. "
            "This is a low edge environment—staying flat is often better than forcing "
            "a trade with no clear directional bias."
        )

    # Generate price series
    start_price = 100 if mode == "Stocks" else 30000
    vol = start_price * 0.005  # ~0.5% volatility
    close = generate_random_walk(n_candles, start_price=start_price, drift=drift * start_price * 0.001, vol=vol)
    o, h, l, c = ohlc_from_close(close, noise=vol * 0.2)

    # Build DataFrame
    if mode == "Stocks":
        tf_minutes = 5
    else:
        tf_minutes = 60

    times = [datetime.now() - timedelta(minutes=tf_minutes * (n_candles - i)) for i in range(n_candles)]
    df = pd.DataFrame({
        "time": times,
        "open": o,
        "high": h,
        "low": l,
        "close": c
    })

    # Indicators
    df["ema9"] = compute_ema(df["close"], 9)
    df["ema20"] = compute_ema(df["close"], 20)
    df["ema50"] = compute_ema(df["close"], 50)
    df["sma200"] = compute_sma(df["close"], 40)  # shorter window for demo

    title = f"{base_title} - {mode}"

    return {
        "df": df,
        "correct_direction": correct_direction,
        "correct_trade": correct_trade,
        "explanation": explanation,
        "title": title,
        "scenario_type": scenario_type,
    }

def plot_candles_with_ma(scenario):
    df = scenario["df"]

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df["time"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="Price"
    ))

    fig.add_trace(go.Scatter(
        x=df["time"], y=df["ema9"],
        mode="lines", name="EMA 9"
    ))
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["ema20"],
        mode="lines", name="EMA 20"
    ))
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["ema50"],
        mode="lines", name="EMA 50"
    ))
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["sma200"],
        mode="lines", name="SMA 200"
    ))

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=40, b=10),
        height=500,
        title=scenario["title"],
    )

    return fig

# ------------------------------
# Streamlit app layout & logic
# ------------------------------

st.set_page_config(page_title="TradeTutor Arena", layout="wide")

st.title("📈 TradeTutor Arena")
st.write(
    "An interactive tutorial game for **stocks & crypto**.\n\n"
    "Each round shows you a chart. You decide the likely move and the best trade."
)

# Sidebar options
st.sidebar.header("Game Settings")
mode = st.sidebar.radio("Market Mode", ["Stocks", "Crypto"], index=0)
difficulty = st.sidebar.radio("Difficulty", ["Beginner", "Intermediate", "Pro"], index=0)

if "scenario" not in st.session_state:
    st.session_state.scenario = make_scenario(mode, difficulty)
    st.session_state.last_mode = mode
    st.session_state.last_diff = difficulty
    st.session_state.result = None

# If user changes mode or difficulty, refresh scenario
if mode != st.session_state.get("last_mode") or difficulty != st.session_state.get("last_diff"):
    st.session_state.scenario = make_scenario(mode, difficulty)
    st.session_state.last_mode = mode
    st.session_state.last_diff = difficulty
    st.session_state.result = None

scenario = st.session_state.scenario

# Show chart
fig = plot_candles_with_ma(scenario)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Your Read on This Chart")

col1, col2 = st.columns(2)

with col1:
    direction_choice = st.radio(
        "1. What is the **most likely next phase**?",
        ["Up", "Down", "Choppy / Sideways"],
        index=0
    )

with col2:
    trade_choice = st.radio(
        "2. What is your **trade decision**?",
        ["Go Long", "Go Short", "Stay Flat"],
        index=0
    )

submitted = st.button("✅ Submit Answer")

if submitted:
    correct_dir = scenario["correct_direction"]
    correct_trade = scenario["correct_trade"]

    dir_correct = (direction_choice == correct_dir)
    trade_correct = (trade_choice == correct_trade)

    if dir_correct and trade_correct:
        verdict = "🔥 **Excellent!** You read the structure and chose the right play."
    elif dir_correct and not trade_correct:
        verdict = (
            "⚠️ Mixed result: you read the **direction** correctly, "
            "but your **trade choice** didn't match the higher-probability play."
        )
    elif not dir_correct and trade_correct:
        verdict = (
            "⚠️ Lucky alignment: your trade choice matches the preferred play, "
            "but your stated bias on direction disagrees with it."
        )
    else:
        verdict = "❌ Not this time. This pattern favors a different bias and trade plan."

    st.session_state.result = {
        "verdict": verdict,
        "direction_choice": direction_choice,
        "trade_choice": trade_choice,
        "correct_direction": correct_dir,
        "correct_trade": correct_trade,
    }

if st.session_state.get("result"):
    result = st.session_state.result
    st.markdown("---")
    st.subheader("Coach’s Feedback")

    st.markdown(result["verdict"])

    col3, col4 = st.columns(2)
    with col3:
        st.markdown(
            f"**Your answers**  \n"
            f"- Direction: `{result['direction_choice']}`  \n"
            f"- Trade: `{result['trade_choice']}`"
        )
    with col4:
        st.markdown(
            f"**Model answers**  \n"
            f"- Direction: `{result['correct_direction']}`  \n"
            f"- Trade: `{result['correct_trade']}`"
        )

    st.info(scenario["explanation"])

st.markdown("---")
if st.button("🔁 New Scenario"):
    st.session_state.scenario = make_scenario(mode, difficulty)
    st.session_state.result = None
    st.experimental_rerun()

st.caption("Tip: change mode or difficulty in the sidebar to get different types of patterns.")
