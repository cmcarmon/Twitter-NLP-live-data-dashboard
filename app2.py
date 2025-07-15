import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
import os
import time
from io import BytesIO
from textblob import TextBlob
import tweepy
from wordcloud import WordCloud
import base64

import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from st_aggrid import AgGrid, GridOptionsBuilder
from streamlit_echarts import st_echarts

# --- THEME CONSTANTS ---
PURPLE_BG = "#F3F0FF"
PURPLE_PALETTE = ["#7B2FF2", "#C3B1E1", "#4B0082", "#A259F7", "#6A0572"]

st.set_page_config(page_title="Live NLP Dashboard", layout="wide", initial_sidebar_state='expanded')
st.title("💜 Live Twitter NLP Dashboard")

# --- Load Bearer Token from Streamlit Secrets Only ---
BEARER_TOKEN = st.secrets.get("TWITTER_BEARER_TOKEN", "")

# --- Session State for Rate Limit Cooldown and Caching ---
if "cooldown_until" not in st.session_state:
    st.session_state["cooldown_until"] = 0
if "selected_tweet_idx" not in st.session_state:
    st.session_state["selected_tweet_idx"] = None
if "tweets_df" not in st.session_state:
    st.session_state["tweets_df"] = None

current_time = time.time()

# --- SIDEBAR CONTROLS (IMPROVED PRESENTATION) ---
with st.sidebar:
    st.markdown("## 💜")
    st.info(
        "- **Enter a hashtag or keyword.**\n"
        "- **Choose number of tweets (10–100).**\n"
        "- **Click 'Fetch Tweets' to analyze.**\n\n"
        "- 'Pissed-offness Metric': +1 = Pissed off, -1 = Amused.\n"
        "- Avoid frequent requests or you'll get a 15-min cooldown."
    )
    query = st.text_input("Keyword/Hashtag", "#python")
    # SLIDER set from 10 to 100 (inclusive), step 1
    tweet_limit = st.slider(
        "Number of Tweets to Fetch", 10, 100, 10, step=1, key="tweet_limit_slider"
    )
    # Disable fetch button during cooldown
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
        "Add your token in Streamlit Secrets Manager as `TWITTER_BEARER_TOKEN`."
    )

# --- Twitter API Setup ---
client = None
if BEARER_TOKEN:
    try:
        client = tweepy.Client(bearer_token=BEARER_TOKEN, wait_on_rate_limit=True)
    except Exception as e:
        st.error(f"Error creating Tweepy client: {e}")

def clean_text(text):
    import re
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[@#]\w+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def liwc_like_categories(text):
    positive_words = set(["great","good","happy","love","fun","cool","amazing","excellent","smile"])
    negative_words = set(["bad","hate","angry","upset","sad","fail","worst","annoy","awful"])
    social_words = set(["friend","team","group","everyone","together","support","we"])
    cognitive_words = set(["think","know","consider","believe","idea","understand","reason"])
    affect_words = set(["love","hate","amaze","fear","angry","joy","sad","happy"])
    words = set(word_tokenize(text.lower()))
    return {
        "Positive Emotion": len(positive_words & words),
        "Negative Emotion": len(negative_words & words),
        "Social": len(social_words & words),
        "Cognitive Processes": len(cognitive_words & words),
        "Affect": len(affect_words & words)
    }

def generate_wordcloud(tweets):
    text = " ".join(tweets)
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(text)
    filtered_text = " ".join([w for w in word_tokens if w.isalpha() and w not in stop_words])
    wc = WordCloud(width=800, height=350, background_color=PURPLE_BG, colormap="plasma").generate(filtered_text)
    buf = BytesIO()
    wc.to_image().save(buf, format="PNG")
    data = base64.b64encode(buf.getvalue()).decode("utf-8")
    return data

@st.cache_data
def fetch_and_analyze(query, tweet_limit):
    tweets, headers, amuse, pissed, times, liwc_scores = [], [], [], [], [], []
    if not BEARER_TOKEN or not client:
        return pd.DataFrame(columns=["Header", "Body", "Timestamp", "Pissed-offness", "Amusement", "LIWC"])
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
                    raw = tweet.text
                    header = (raw[:60] + "...") if len(raw) > 60 else raw
                    body = clean_text(raw)
                    polarity = TextBlob(body).sentiment.polarity
                    amuse_score = round(-polarity, 3)
                    pissed_score = round(polarity * -1, 3)  # +1 = pissed off, -1 = amused
                    scores = liwc_like_categories(body)
                    tweets.append(raw)
                    headers.append(header)
                    amuse.append(amuse_score)
                    pissed.append(pissed_score)
                    times.append(tweet.created_at)
                    liwc_scores.append(scores)
    except tweepy.TooManyRequests:
        st.session_state["cooldown_until"] = time.time() + 15 * 60  # 15 min cooldown
        st.error("Twitter API rate limit exceeded. Please wait 15 minutes before trying again.")
    except Exception as e:
        st.error(f"Error fetching tweets: {e}")
    df = pd.DataFrame({
        "Header": headers,
        "Body": tweets,
        "Timestamp": times,
        "Pissed-offness": pissed,
        "Amusement": amuse,
        "LIWC": liwc_scores
    })
    return df

def display_metrics(df):
    avg_pissed = df["Pissed-offness"].mean() if not df.empty else 0
    avg_pissed_pct = abs(avg_pissed) * 100
    # Main metric card logic
    if avg_pissed >= 0:
        metric_label = f"{avg_pissed_pct:.0f}% Pissed off"
        explanation = "(+100 = maximum pissed off, -100 = maximum amused, 0 = neutral)"
    else:
        metric_label = f"{avg_pissed_pct:.0f}% Amused"
        explanation = "(+100 = maximum pissed off, -100 = maximum amused, 0 = neutral)"
    # Classify tweets
    num_amused = (df["Pissed-offness"] < -0.05).sum()
    num_pissed = (df["Pissed-offness"] > 0.05).sum()
    amused_pct = 100 * num_amused / len(df)
    pissed_pct = 100 * num_pissed / len(df)
    neutral_pct = 100 - amused_pct - pissed_pct
    c1, c2, c3 = st.columns(3)
    c1.metric("Pissed-offness metric", metric_label, help=explanation)
    c2.metric("Amused Tweets", f"{amused_pct:.1f}%")
    c3.metric("Pissed Off Tweets", f"{pissed_pct:.1f}%")

def display_dashboard(df):
    # Interactive grid table for tweet selection
    gb = GridOptionsBuilder.from_dataframe(df[["Header", "Pissed-offness", "Amusement", "Timestamp"]])
    gb.configure_selection(selection_mode="single", use_checkbox=False)
    gb.configure_pagination()
    grid_options = gb.build()
    ag_grid = AgGrid(df[["Header", "Pissed-offness", "Amusement", "Timestamp"]],
                     gridOptions=grid_options, height=290, theme="material")
    if ag_grid['selected_rows']:
        idx = ag_grid['selected_rows'][0]['_selectedRowNodeInfo']['nodeRowIndex']
        row = df.iloc[idx]
    else:
        row = df.iloc[0]
    # Show tweet details
    st.markdown(f"**{row['Header']}**")
    st.markdown(f"<span style='font-size:0.92em'>{clean_text(row['Body'])}</span>", unsafe_allow_html=True)
    st.caption(f"🗓️ {row['Timestamp']} | Pissed-offness: {row['Pissed-offness']:+.2f} | Amusement: {row['Amusement']:+.2f}")
    # Metrics block
    display_metrics(df)
    # LIWC pie chart
    feature_obj = [{"value": v, "name": k} for k, v in row['LIWC'].items()]
    st_echarts({
        "title": {"text": "LIWC-Style Theme Breakdown", "left": "center"},
        "tooltip": {},
        "legend": {"top": "bottom"},
        "series": [
            {
                "name": "Category",
                "type": "pie",
                "radius": ["30%", "60%"],
                "roseType": "area",
                "data": feature_obj,
                "label": {"show": True, "fontSize": 16},
            }
        ],
    }, height="350px")

    # Timeline chart
    line_chart = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X('Timestamp:T', title="Time", axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('Pissed-offness:Q', title="Pissed-offness"),
        tooltip=["Header", "Pissed-offness", "Amusement", "Timestamp"]
    ).properties(width=720, title="Pissed-offness Over Time")
    st.altair_chart(line_chart, use_container_width=True)

    # Word Cloud
    st.markdown("### Word Cloud Overview")
    wordcloud_img = generate_wordcloud([clean_text(t) for t in df["Body"]])
    st.image(f"data:image/png;base64,{wordcloud_img}", use_column_width=True, caption="Words across tweets")

    # Download cached data
    csv_data = df.to_csv(index=False)
    st.download_button("Download Session Tweets (CSV)", csv_data, file_name="twitter_nlp_dashboard.csv")

if fetch_button:
    if current_time < st.session_state["cooldown_until"]:
        wait = int(st.session_state["cooldown_until"] - current_time)
        st.warning(f"Rate limit in effect. Please wait {wait//60} min {wait%60} sec before trying again.")
    else:
        df = fetch_and_analyze(query, tweet_limit)
        st.session_state['tweets_df'] = df  # Cache last fetch

        if not BEARER_TOKEN:
            st.warning("Bearer Token missing—live data fetch won't work until added in Streamlit Cloud's Secrets.")
        elif df.empty:
            st.warning("No tweets found. Try a popular query like #news or wait for new tweets.")
        else:
            st.subheader("🟣 Live Tweets")
            display_dashboard(df)
else:
    # If we have cached data, allow visualization without API requests
    if st.session_state.get('tweets_df') is not None and not st.session_state['tweets_df'].empty:
        st.info("🔁 Showing last fetched tweets from session cache. No API request used.")
        display_dashboard(st.session_state['tweets_df'])
    else:
        st.info("Click 'Fetch Tweets' in the sidebar to load Twitter data.")
