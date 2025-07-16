import os
import streamlit as st
import pandas as pd
import plotly.express as px
import time
import base64
from io import BytesIO
from textblob import TextBlob
import tweepy
from wordcloud import WordCloud
import nltk

# --- NLTK Download Setup ---
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
if nltk_data_dir not in nltk.data.path:
    nltk.data.path.insert(0, nltk_data_dir)
nltk.download("punkt", download_dir=nltk_data_dir, quiet=True)
nltk.download("punkt_tab", download_dir=nltk_data_dir, quiet=True)
nltk.download("stopwords", download_dir=nltk_data_dir, quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

PURPLE_BG = "#F3F0FF"
PURPLE_PALETTE = ["#7B2FF2", "#C3B1E1", "#4B0082", "#A259F7", "#6A0572"]
DARK_TEXT = "#2D1A47"

st.set_page_config(page_title="Live NLP Dashboard", layout="wide")
st.title("💜 Live Twitter NLP Dashboard")

BEARER_TOKEN = st.secrets.get("TWITTER_BEARER_TOKEN", "")

if "tweets_df" not in st.session_state:
    st.session_state["tweets_df"] = None
if "cooldown_until" not in st.session_state:
    st.session_state["cooldown_until"] = 0

current_time = time.time()

with st.sidebar:
    st.markdown("## 💜")
    st.info(
        "- **Enter a hashtag or keyword.**\n"
        "- **Choose number of tweets (10–100).**\n"
        "- Tweets shown in dropdown; select for analysis.\n"
        "- 'Pissed-offness': +1 = Pissed off, -1 = Amused.\n"
        "- Fetch infrequently to avoid Twitter API cooldown (15 min after limit)."
    )
    query = st.text_input("Keyword/Hashtag", "#python")
    tweet_limit = st.slider(
        "Number of Tweets to Fetch", 10, 100, 10, step=1, key="tweet_limit_slider"
    )
    if current_time < st.session_state["cooldown_until"]:
        wait = int(st.session_state["cooldown_until"] - current_time)
        fetch_button = st.button("Fetch Tweets", disabled=True)
        st.info(f"Rate limit active. Please wait {wait // 60} min {wait % 60} sec.")
    else:
        fetch_button = st.button("Fetch Tweets")

if not BEARER_TOKEN:
    st.warning(
        "Twitter Bearer Token not found. Add your token in Streamlit Secrets Manager as `TWITTER_BEARER_TOKEN`."
    )

client = None
if BEARER_TOKEN:
    try:
        client = tweepy.Client(bearer_token=BEARER_TOKEN)
    except Exception as e:
        st.error(f"Error creating Tweepy client: {e}")

def clean_text(text):
    import re
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[@#]\w+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def liwc_like_categories(text):
    positive_words = set(["great", "good", "happy", "love", "fun", "cool", "amazing", "excellent", "smile"])
    negative_words = set(["bad", "hate", "angry", "upset", "sad", "fail", "worst", "annoy", "awful"])
    social_words   = set(["friend", "team", "group", "everyone", "together", "support", "we"])
    cognitive_words= set(["think", "know", "consider", "believe", "idea", "understand", "reason"])
    affect_words   = set(["love", "hate", "amaze", "fear", "angry", "joy", "sad", "happy"])
    words = set(word_tokenize(str(text).lower()))
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
def fetch_and_cache_tweets(query, tweet_limit):
    tweets = []
    timestamps = []
    if not BEARER_TOKEN or not client:
        return pd.DataFrame()
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
                    tweets.append(tweet.text)
                    timestamps.append(tweet.created_at)
    except tweepy.errors.TooManyRequests as e:
        # Get x-rate-limit-reset if available; else set 15 min from now
        headers = getattr(e, "response", None)
        reset_ts = 0
        if headers and hasattr(headers, "headers"):
            reset_ts = int(headers.headers.get("x-rate-limit-reset", 0))
        wait_sec = max(reset_ts - time.time(), 900) if reset_ts else 900
        st.session_state["cooldown_until"] = time.time() + wait_sec
        st.warning(f"Twitter API rate limit exceeded. Please wait {int(wait_sec // 60)} min {int(wait_sec % 60)} sec before trying again.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching tweets: {e}")
        return pd.DataFrame()
    df = pd.DataFrame({'Tweet': tweets, 'Timestamp': timestamps})
    return df

def display_metrics(pissed_scores):
    if len(pissed_scores) > 0:
        num_amused = sum(1 for x in pissed_scores if x < -0.05)
        num_pissed = sum(1 for x in pissed_scores if x > 0.05)
        amused_pct = 100 * num_amused / len(pissed_scores)
        pissed_pct = 100 * num_pissed / len(pissed_scores)
    else:
        amused_pct = pissed_pct = 0
    avg_pissed = sum(pissed_scores) / len(pissed_scores) if len(pissed_scores) else 0
    avg_pct = abs(avg_pissed) * 100
    if avg_pissed >= 0:
        metric_label = f"{avg_pct:.0f}% Pissed off"
        explanation = "(+100 = max pissed off, -100 = max amused, 0 = neutral)"
    else:
        metric_label = f"{avg_pct:.0f}% Amused"
        explanation = "(+100 = max pissed off, -100 = max amused, 0 = neutral)"
    c1, c2, c3 = st.columns(3)
    c1.metric("Pissed-offness metric", metric_label, help=explanation)
    c2.metric("Amused Tweets", f"{amused_pct:.1f}%")
    c3.metric("Pissed Off Tweets", f"{pissed_pct:.1f}%")

def display_dashboard(df):
    if df.empty:
        st.info("No tweets found for your query. Try a different keyword or wait for new tweets.")
        return
    st.write("#### Fetched Tweets")
    df['Short Tweet'] = df["Tweet"].apply(lambda t: t[:70] + ("..." if len(t) > 70 else ""))
    tweet_choices = [(i, df.iloc[i]['Short Tweet']) for i in range(len(df))]
    selected_idx = st.selectbox(
        "Select a tweet to analyze:",
        tweet_choices,
        format_func=lambda pair: f"{pair[0]+1}: {pair[1]}"
    )[0]
    selected_row = df.iloc[selected_idx]
    selected_text = selected_row["Tweet"]
    selected_time = selected_row["Timestamp"]

    # NLP Analysis (bulk for overall metrics, per-tweet for visuals)
    pissed_scores = []
    amuse_scores = []
    liwc_list = []
    for tweet in df["Tweet"]:
        polarity = TextBlob(tweet).sentiment.polarity
        pissed_score = round(-polarity, 3)
        amuse_score = round(polarity, 3)
        liwc = liwc_like_categories(tweet)
        pissed_scores.append(pissed_score)
        amuse_scores.append(amuse_score)
        liwc_list.append(liwc)
    # Metrics for all tweets
    display_metrics(pissed_scores)
    # Selected tweet analysis
    st.markdown(f"**{selected_row['Short Tweet']}**")
    st.code(selected_text, language="markdown")
    st.caption(f"🗓️ {selected_time} | Pissed-offness: {round(-TextBlob(selected_text).sentiment.polarity,2):+}")

    st.write("##### LIWC-style Feature Analysis (selected tweet):")
    sel_liwc = liwc_like_categories(selected_text)
    sel_liwc_df = pd.DataFrame(list(sel_liwc.items()), columns=["Category", "Count"])
    sel_liwc_fig = px.bar(sel_liwc_df,
        x="Category", y="Count", color="Count", color_continuous_scale=PURPLE_PALETTE,
        title="LIWC-style Features for Selected Tweet"
    )
    sel_liwc_fig.update_layout(
        plot_bgcolor=PURPLE_BG,
        paper_bgcolor=PURPLE_BG,
        font_color=DARK_TEXT,
        title_font_color=DARK_TEXT,
        xaxis_title_font=dict(color=DARK_TEXT),
        yaxis_title_font=dict(color=DARK_TEXT)
    )
    st.plotly_chart(sel_liwc_fig, use_container_width=True)

    st.markdown("##### Overall LIWC Feature Summary")
    all_liwc_summary = pd.DataFrame(liwc_list).sum().reset_index()
    all_liwc_summary.columns = ["Category", "Total Count"]
    liwc_fig = px.bar(
        all_liwc_summary,
        x="Category",
        y="Total Count",
        color="Total Count",
        color_continuous_scale=PURPLE_PALETTE,
        title="LIWC-style Psychological Categories (All Tweets)"
    )
    liwc_fig.update_layout(
        plot_bgcolor=PURPLE_BG,
        paper_bgcolor=PURPLE_BG,
        font_color=DARK_TEXT,
        title_font_color=DARK_TEXT,
        xaxis_title_font=dict(color=DARK_TEXT),
        yaxis_title_font=dict(color=DARK_TEXT)
    )
    st.plotly_chart(liwc_fig, use_container_width=True)

    st.markdown("### Word Cloud Overview")
    wordcloud_img = generate_wordcloud([str(t) for t in df["Tweet"]])
    st.image(f"data:image/png;base64,{wordcloud_img}", caption="Words across tweets")

    metric_df = df.copy()
    metric_df["Pissed-offness"] = pissed_scores
    metric_trend = px.line(
        metric_df,
        x="Timestamp",
        y="Pissed-offness",
        title="Pissed-offness Metric Trend",
        markers=True,
        color_discrete_sequence=PURPLE_PALETTE
    )
    metric_trend.update_layout(
        plot_bgcolor=PURPLE_BG,
        paper_bgcolor=PURPLE_BG,
        font_color=DARK_TEXT,
        title_font_color=DARK_TEXT,
        xaxis_title_font=dict(color=DARK_TEXT),
        yaxis_title_font=dict(color=DARK_TEXT)
    )
    st.plotly_chart(metric_trend, use_container_width=True)

    csv_data = df.to_csv(index=False)
    st.download_button("Download Session Tweets (CSV)", csv_data, file_name="twitter_nlp_dashboard.csv")

if fetch_button:
    if current_time < st.session_state["cooldown_until"]:
        wait = int(st.session_state["cooldown_until"] - current_time)
        st.warning(f"Rate limit in effect. Please wait {wait // 60} min {wait % 60} sec before trying again.")
    else:
        df = fetch_and_cache_tweets(query, tweet_limit)
        if not df.empty:
            st.session_state['tweets_df'] = df
            st.session_state["cooldown_until"] = 0
        if not BEARER_TOKEN:
            st.warning("Bearer Token missing—live data fetch won't work until the token is added in Streamlit Cloud's Secrets.")
        elif df is None or df.empty:
            st.warning("No suitable tweets found. Try a popular keyword or increase tweet limit.")
        else:
            st.subheader("🟣 Live Tweets (Dropdown Selection)")
            display_dashboard(df)
else:
    df = st.session_state.get('tweets_df')
    if df is not None and not df.empty:
        st.info("🔁 Reviewing last fetched tweets from session cache. No API request is used.")
        display_dashboard(df)
    else:
        st.info("Click 'Fetch Tweets' in the sidebar to load Twitter data.")
