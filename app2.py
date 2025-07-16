import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
import time
import base64
from io import BytesIO
from textblob import TextBlob
import tweepy
from wordcloud import WordCloud

# --- NLTK Download for required NLP resources ---
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

PURPLE_BG = "#F3F0FF"
PURPLE_PALETTE = ["#7B2FF2", "#C3B1E1", "#4B0082", "#A259F7", "#6A0572"]

st.set_page_config(page_title="Live NLP Dashboard", layout="wide")
st.title("💜 Live Twitter NLP Dashboard")

BEARER_TOKEN = st.secrets.get("TWITTER_BEARER_TOKEN", "")

if "cooldown_until" not in st.session_state:
    st.session_state["cooldown_until"] = 0
if "tweets_df" not in st.session_state:
    st.session_state["tweets_df"] = None

current_time = time.time()

with st.sidebar:
    st.markdown("## 💜")
    st.info(
        "- **Enter a hashtag or keyword.**\n"
        "- **Choose number of tweets (10–100).**\n"
        "- **Click 'Fetch Tweets' to analyze.**\n"
        "- 'Pissed-offness': +1 = Pissed off, -1 = Amused.\n"
        "- Avoid frequent requests to prevent a 15-min cooldown."
    )
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

if not BEARER_TOKEN:
    st.warning(
        "Twitter Bearer Token not found. Add your token in Streamlit Secrets Manager as `TWITTER_BEARER_TOKEN`."
    )

client = None
if BEARER_TOKEN:
    try:
        client = tweepy.Client(bearer_token=BEARER_TOKEN, wait_on_rate_limit=False)
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
    social_words   = set(["friend","team","group","everyone","together","support","we"])
    cognitive_words= set(["think","know","consider","believe","idea","understand","reason"])
    affect_words   = set(["love","hate","amaze","fear","angry","joy","sad","happy"])
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
def fetch_and_analyze(query, tweet_limit):
    tweets, pissed, amuse, times, liwc_scores = [], [], [], [], []
    if not BEARER_TOKEN or not client:
        return pd.DataFrame()
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
                    polarity = TextBlob(text).sentiment.polarity
                    pissed_score = round(-polarity, 3)
                    amuse_score = round(polarity, 3)
                    liwc = liwc_like_categories(text)
                    tweets.append(text)
                    pissed.append(pissed_score)
                    amuse.append(amuse_score)
                    liwc_scores.append(liwc)
                    times.append(tweet.created_at)
    except tweepy.TooManyRequests:
        st.session_state["cooldown_until"] = time.time() + 15 * 60
        st.warning("Twitter API rate limit exceeded. Please wait 15 minutes before trying again. You can review cached data below.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching tweets: {e}")
        return pd.DataFrame()
    df = pd.DataFrame({
        "Tweet": tweets,
        "Timestamp": times,
        "Pissed-offness": pissed,
        "Amusement": amuse,
        "LIWC": liwc_scores
    })
    return df

def display_metrics(df):
    avg_pissed = df["Pissed-offness"].mean() if not df.empty else 0
    avg_pct = abs(avg_pissed) * 100
    if avg_pissed >= 0:
        metric_label = f"{avg_pct:.0f}% Pissed off"
        explanation = "(+100 = max pissed off, -100 = max amused, 0 = neutral)"
    else:
        metric_label = f"{avg_pct:.0f}% Amused"
        explanation = "(+100 = max pissed off, -100 = max amused, 0 = neutral)"
    num_amused = (df["Pissed-offness"] < -0.05).sum()
    num_pissed = (df["Pissed-offness"] > 0.05).sum()
    amused_pct = 100 * num_amused / len(df) if len(df) > 0 else 0
    pissed_pct = 100 * num_pissed / len(df) if len(df) > 0 else 0
    c1, c2, c3 = st.columns(3)
    c1.metric("Pissed-offness metric", metric_label, help=explanation)
    c2.metric("Amused Tweets", f"{amused_pct:.1f}%")
    c3.metric("Pissed Off Tweets", f"{pissed_pct:.1f}%")

def display_dashboard(df):
    st.write("#### Tweets (click for details)")
    df['Short Tweet'] = df["Tweet"].apply(lambda t: t[:70] + ("..." if len(t) > 70 else ""))
    selected = st.selectbox("Select a tweet to see details and LIWC features:",
                            range(len(df)), format_func=lambda i: df.iloc[i]["Short Tweet"] if not df.empty else "")
    row = df.iloc[selected] if len(df) > 0 else None
    display_metrics(df)

    if row is not None:
        st.markdown(f"**{row['Short Tweet']}**")
        st.code(row["Tweet"], language="markdown")
        st.caption(f"🗓️ {row['Timestamp']} | Pissed-offness: {row['Pissed-offness']:+.2f}")
        st.write("##### LIWC-style Feature Analysis:")
        feat_df = pd.DataFrame(list(row['LIWC'].items()), columns=["Category", "Count"])
        liwc_fig = px.bar(feat_df, x="Category", y="Count", color="Count", color_continuous_scale=PURPLE_PALETTE,
                          title="LIWC-style Psychological Categories")
        liwc_fig.update_layout(plot_bgcolor=PURPLE_BG, paper_bgcolor=PURPLE_BG, font_color="#2D1A47")
        st.plotly_chart(liwc_fig, use_container_width=True)

        st.write("##### 3D Visualization: Tweet # vs Pissed-offness vs Time")
        z_time = pd.to_datetime(df["Timestamp"]).astype(int) // 10**9
        fig3d = go.Figure(data=[go.Scatter3d(
            x=list(range(len(df))),
            y=df["Pissed-offness"],
            z=z_time,
            mode="markers",
            marker=dict(size=7, color=df["Pissed-offness"], colorscale=PURPLE_PALETTE, opacity=0.8),
            text=df["Short Tweet"]
        )])
        fig3d.update_layout(
            scene=dict(xaxis_title="Tweet #", yaxis_title="Pissed-offness", zaxis_title="Timestamp"),
            paper_bgcolor=PURPLE_BG,
            font=dict(color="#2D1A47"), width=800, height=460
        )
        st.plotly_chart(fig3d, use_container_width=True)

    st.markdown("### Word Cloud Overview")
    wordcloud_img = generate_wordcloud([str(t) for t in df["Tweet"]])
    st.image(f"data:image/png;base64,{wordcloud_img}", use_column_width=True, caption="Words across tweets")

    trend = px.line(df, x="Timestamp", y="Pissed-offness", title="Pissed-offness Metric Trend",
                    markers=True, color_discrete_sequence=PURPLE_PALETTE)
    trend.update_layout(plot_bgcolor=PURPLE_BG, paper_bgcolor=PURPLE_BG, font_color="#2D1A47")
    st.plotly_chart(trend, use_container_width=True)

    csv_data = df.to_csv(index=False)
    st.download_button("Download Session Tweets (CSV)", csv_data, file_name="twitter_nlp_dashboard.csv")

if fetch_button:
    if current_time < st.session_state["cooldown_until"]:
        wait = int(st.session_state["cooldown_until"] - current_time)
        st.warning(f"Rate limit in effect. Please wait {wait//60} min {wait%60} sec before trying again.")
    else:
        df = fetch_and_analyze(query, tweet_limit)
        if not df.empty:
            st.session_state['tweets_df'] = df  # Cache last fetch
            st.session_state["cooldown_until"] = 0  # Clear cooldown after success
        if not BEARER_TOKEN:
            st.warning("Bearer Token missing—live data fetch won't work until the token is added in Streamlit Cloud's Secrets.")
        elif df is None or df.empty:
            st.warning("No tweets found. Try a popular query like #news or wait for new tweets.")
        else:
            st.subheader("🟣 Live Tweets")
            display_dashboard(df)
else:
    df = st.session_state.get('tweets_df')
    if df is not None and not df.empty:
        st.info("🔁 Reviewing last fetched tweets from session cache. No API request is used.")
        display_dashboard(df)
    else:
        st.info("Click 'Fetch Tweets' in the sidebar to load Twitter data.")
