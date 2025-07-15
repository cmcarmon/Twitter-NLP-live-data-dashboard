import streamlit as st
import pandas as pd
import plotly.express as px
import os
import time
from textblob import TextBlob
import tweepy

# --- THEME CONSTANTS ---
PURPLE_BG = "#F3F0FF"
PURPLE_PALETTE = ["#7B2FF2", "#C3B1E1", "#4B0082", "#A259F7", "#6A0572"]

st.set_page_config(page_title="Live NLP Dashboard", layout="wide")
st.title("💜 Live Twitter NLP Dashboard")

# --- Load Bearer Token from Streamlit Secrets Only ---
BEARER_TOKEN = st.secrets.get("TWITTER_BEARER_TOKEN", "")

# --- Session State for Rate Limit Cooldown ---
if "cooldown_until" not in st.session_state:
    st.session_state["cooldown_until"] = 0

current_time = time.time()

# --- SIDEBAR CONTROLS (CLEANER PRESENTATION) ---
with st.sidebar:
    st.markdown("## 💜")

    # Friendly, unobtrusive instructions (improved formatting)
    st.info(
        "Use the controls below to search and analyze live Twitter sentiment:\n\n"
        "- Enter a hashtag or keyword.\n"
        "- Pick the number of tweets to fetch (10–100).\n"
        "- Click 'Fetch Tweets' to load data and see results.\n\n"
        "**Tip:** Twitter API rate limits are strict. Avoid frequent searches to prevent a 15-minute cooldown."
    )

    # Search bar above button/sliders for more intuitive UX
    query = st.text_input("Keyword/Hashtag", "#python")

    tweet_limit = st.slider(
        "Number of Tweets to Fetch", 10, 100, 10, step=1, key="tweet_limit_slider"
    )

    if current_time < st.session_state["cooldown_until"]:
        wait = int(st.session_state["cooldown_until"] - current_time)
        fetch_button = st.button("Fetch Tweets", disabled=True)
        st.info(f"Rate limit active. Please wait {wait//60} min {wait%60} sec.")
    else:
        fetch_button = st.button("Fetch Tweets")

# --- Bearer Token Warning (Non-Fatal) ---
if not BEARER_TOKEN:
    st.warning(
        "**Twitter Bearer Token not found.**\n"
        "Add your token in Streamlit Cloud's Secrets Manager as `TWITTER_BEARER_TOKEN`.\n"
        "You can still use the UI, but live data cannot be loaded."
    )

# --- Twitter API Setup ---
client = None
if BEARER_TOKEN:
    try:
        client = tweepy.Client(bearer_token=BEARER_TOKEN, wait_on_rate_limit=True)
    except Exception as e:
        st.error(f"Error creating Tweepy client: {e}")

# --- Caching: Only Accept Hashable Params ---
@st.cache_data
def fetch_and_analyze(query, tweet_limit):
    tweets, sentiments, times = [], [], []
    if not BEARER_TOKEN or not client:
        return pd.DataFrame(columns=["Timestamp", "Tweet", "Sentiment"])
    # Defensive: ensure tweet_limit always within correct bounds
    tweet_limit = max(10, min(tweet_limit, 100))
    try:
        response = client.search_recent_tweets(
            query=query,
            max_results=tweet_limit,
            tweet_fields=["created_at", "lang"]
        )
        if response.data is not None:
            for tweet in response.data:
                lang = getattr(tweet, "lang", None)
                if lang is None or lang == "en":
                    text = tweet.text
                    sentiment = TextBlob(text).sentiment.polarity
                    tweets.append(text)
                    sentiments.append(sentiment)
                    times.append(tweet.created_at)
    except tweepy.TooManyRequests:
        st.session_state["cooldown_until"] = time.time() + 15 * 60  # 15 min cooldown
        st.error("Twitter API rate limit exceeded. Please wait 15 minutes before trying again.")
    except Exception as e:
        st.error(f"Error fetching tweets: {e}")
    return pd.DataFrame({"Timestamp": times, "Tweet": tweets, "Sentiment": sentiments})

# --- Main Dashboard Logic ---
if fetch_button:
    if current_time < st.session_state["cooldown_until"]:
        wait = int(st.session_state["cooldown_until"] - current_time)
        st.warning(f"Rate limit in effect. Please wait {wait//60} min {wait%60} sec before trying again.")
    else:
        df = fetch_and_analyze(query, tweet_limit)
        if not BEARER_TOKEN:
            st.warning("Bearer Token missing—live data fetch won't work until the token is added in Streamlit Cloud's Secrets.")
        elif df.empty:
            st.warning("No tweets found. Try a popular query like #news or wait for new tweets.")
        else:
            st.subheader("🟣 Live Tweets")
            st.dataframe(df[["Timestamp", "Tweet", "Sentiment"]], use_container_width=True, height=260)

            avg_sentiment = df["Sentiment"].mean() if not df.empty else 0
            pos_count = (df["Sentiment"] > 0).sum()
            neg_count = (df["Sentiment"] < 0).sum()
            k1, k2, k3 = st.columns(3)
            k1.metric("Avg Sentiment", f"{avg_sentiment:.2f}")
            k2.metric("Positive Tweets", int(pos_count))
            k3.metric("Negative Tweets", int(neg_count))

            fig1 = px.line(
                df, x="Timestamp", y="Sentiment", title="Sentiment Over Time",
                markers=True, color_discrete_sequence=PURPLE_PALETTE
            )
            fig1.update_layout(
                plot_bgcolor=PURPLE_BG, paper_bgcolor=PURPLE_BG, font_color="#2D1A47",
                font_family="sans-serif", title_font_color="#7B2FF2"
            )
            st.plotly_chart(fig1, use_container_width=True)

            fig2 = px.histogram(
                df, x="Sentiment", nbins=10, title="Sentiment Distribution",
                color_discrete_sequence=PURPLE_PALETTE
            )
            fig2.update_layout(
                plot_bgcolor=PURPLE_BG, paper_bgcolor=PURPLE_BG, font_color="#2D1A47",
                font_family="sans-serif", title_font_color="#7B2FF2"
            )
            st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Click 'Fetch Tweets' in the sidebar to load Twitter data.")
