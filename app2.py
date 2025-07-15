import streamlit as st
import pandas as pd
import plotly.express as px
import os
from textblob import TextBlob
import tweepy
import dotenv

# --- THEME CONSTANTS ---
PURPLE_BG = "#F3F0FF"
PURPLE_PALETTE = ["#7B2FF2", "#C3B1E1", "#4B0082", "#A259F7", "#6A0572"]

# --- LOAD THE .env FILE & TWITTER API TOKEN ---
dotenv.load_dotenv()
BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
if not BEARER_TOKEN:
    st.error("Twitter Bearer Token not found. Make sure your .env is in the project root and includes 'TWITTER_BEARER_TOKEN=...'.")

# --- TWITTER API SETUP ---
client = tweepy.Client(bearer_token=BEARER_TOKEN, wait_on_rate_limit=True)

st.set_page_config(page_title="Live NLP Dashboard", layout="wide")
st.title("💜 Live Twitter NLP Dashboard")

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.markdown("### Theme: Purple/Indigo 💜")
    st.markdown(
        f"<div style='background-color:{PURPLE_BG}; padding:10px; border-radius:10px; color:#2D1A47;'>"
        "Enter a hashtag or keyword to analyze Twitter in real time."
        "</div>", unsafe_allow_html=True
    )
    query = st.text_input("Keyword/Hashtag", "#python")
    tweet_limit = st.slider("Number of Tweets to Fetch", 1, 20, 3, key="tweet_limit_slider")  # default 3
    fetch_button = st.button("Fetch Tweets")

@st.cache_data
def fetch_and_analyze(query, tweet_limit):
    tweets, sentiments, times = [], [], []
    try:
        response = client.search_recent_tweets(
            query=query,
            max_results=min(tweet_limit, 100),
            tweet_fields=["created_at", "lang"]
        )
        if response.data is not None:
            for tweet in response.data:
                if tweet.lang == "en":
                    text = tweet.text
                    sentiment = TextBlob(text).sentiment.polarity
                    tweets.append(text)
                    sentiments.append(sentiment)
                    times.append(tweet.created_at)
    except Exception as e:
        st.error(f"Error fetching tweets: {e}")
    return pd.DataFrame({"Timestamp": times, "Tweet": tweets, "Sentiment": sentiments})

if fetch_button:
    if not BEARER_TOKEN:
        st.stop()
    df = fetch_and_analyze(query, tweet_limit)
    if df.empty:
        st.warning("No tweets found. Try another keyword or wait for new tweets.")
    else:
        st.subheader("🟣 Live Tweets")
        st.dataframe(df[["Timestamp", "Tweet", "Sentiment"]], use_container_width=True, height=260)

        avg_sentiment = df["Sentiment"].mean()
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
    st.info("Click 'Fetch Tweets' in the sidebar to load the latest Twitter data.")
