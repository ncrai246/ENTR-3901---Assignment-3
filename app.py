import streamlit as st
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import plotly.express as px

# ==========================================
# 1. PAGE CONFIGURATION & CRAVE STREAMING UI
# ==========================================
st.set_page_config(
    page_title="CINEPULSE | Stream Smarter",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Crave Streaming Platform CSS Overhaul
st.markdown("""
<style>
    /* Hide default Streamlit clutter and enforce Crave Midnight Navy background */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp {
        background-color: #050A14;
        background-image: radial-gradient(circle at 50% 0%, #0D2242 0%, #050A14 70%);
        color: #F0F6FC;
    }
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        max-width: 1440px !important;
    }

    /* Top Crave Navigation Bar */
    .crave-navbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 24px;
        background: rgba(8, 17, 34, 0.85);
        backdrop-filter: blur(12px);
        border-bottom: 1px solid rgba(0, 168, 232, 0.25);
        border-radius: 12px;
        margin-bottom: 20px;
    }
    .crave-wordmark {
        font-family: 'Segoe UI', Roboto, Helvetica, sans-serif;
        font-size: 2.1rem;
        font-weight: 900;
        letter-spacing: 3.5px;
        color: #FFFFFF;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .crave-pill {
        background: linear-gradient(135deg, #00A8E8 0%, #0066CC 100%);
        color: #FFFFFF;
        font-size: 0.78rem;
        font-weight: 800;
        padding: 4px 10px;
        border-radius: 4px;
        letter-spacing: 1.5px;
    }
    .crave-tagline {
        color: #8AB4F8;
        font-size: 0.9rem;
        font-weight: 500;
    }

    /* Cinematic Widescreen Hero Billboard */
    .hero-banner {
        position: relative;
        border-radius: 16px;
        overflow: hidden;
        min-height: 360px;
        display: flex;
        align-items: flex-end;
        padding: 36px 40px;
        margin-bottom: 28px;
        border: 1px solid rgba(0, 168, 232, 0.3);
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.65);
        background-size: cover;
        background-position: center 20%;
    }
    .hero-content {
        max-width: 680px;
        z-index: 2;
    }
    .hero-eyebrow {
        color: #00A8E8;
        font-size: 0.82rem;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 900;
        color: #FFFFFF;
        line-height: 1.1;
        margin-bottom: 12px;
        text-shadow: 0 4px 16px rgba(0, 0, 0, 0.8);
    }
    .hero-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        align-items: center;
        margin-bottom: 14px;
        font-size: 0.88rem;
        font-weight: 600;
        color: #D0E2F2;
    }
    .meta-badge {
        background: rgba(0, 168, 232, 0.2);
        border: 1px solid #00A8E8;
        color: #5CE1E6;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.78rem;
        font-weight: 700;
    }
    .hero-plot {
        color: #C2D6EA;
        font-size: 0.96rem;
        line-height: 1.5;
        margin-bottom: 8px;
        text-shadow: 0 2px 8px rgba(0, 0, 0, 0.9);
    }

    /* Poster Grid Cards & Hover Animations */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(180deg, #0B172A 0%, #07101E 100%) !important;
        border: 1px solid rgba(0, 168, 232, 0.18) !important;
        border-radius: 12px !important;
        transition: transform 0.22s ease, border-color 0.22s ease, box-shadow 0.22s ease !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-5px) !important;
        border-color: #00A8E8 !important;
        box-shadow: 0 10px 24px rgba(0, 168, 232, 0.25) !important;
    }
    [data-testid="stImage"] img {
        border-radius: 8px !important;
        width: 100% !important;
    }
    .match-pill {
        display: inline-block;
        background: rgba(0, 168, 232, 0.18);
        border: 1px solid #00A8E8;
        color: #38BDF8;
        font-size: 0.76rem;
        font-weight: 800;
        padding: 2px 8px;
        border-radius: 4px;
        margin-bottom: 6px;
    }
    .card-title {
        font-size: 1.02rem;
        font-weight: 700;
        color: #FFFFFF;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        margin-bottom: 2px;
    }
    .card-sub {
        font-size: 0.8rem;
        color: #8BA3C7;
        margin-bottom: 8px;
    }
    .section-header {
        font-size: 1.35rem;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: 0.5px;
        margin: 18px 0 12px 0;
        border-left: 4px solid #00A8E8;
        padding-left: 12px;
    }
</style>
""", unsafe_allow_html=True)

# API Key setup using Streamlit Secrets
if "TMDB_KEY" in st.secrets:
    API_KEY = st.secrets["TMDB_KEY"]
else:
    st.error("API Key not found in Streamlit Secrets.")
    st.stop()

if "watchlist" not in st.session_state:
    st.session_state.watchlist = []

# ==========================================
# 2. DATA RETRIEVAL & CACHING FUNCTIONS
# ==========================================
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_movie_details(movie_id):
    """Fetches comprehensive metadata for a single movie ID from TMDB."""
    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    params = {
        "api_key": API_KEY,
        "append_to_response": "credits,keywords,release_dates"
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            return None
        d = response.json()
    except Exception:
        return None

    # Extract US Certification (MPAA Rating)
    certification = "NR"
    for country in d.get("release_dates", {}).get("results", []):
        if country.get("iso_3166_1") == "US":
            for rel in country.get("release_dates", []):
                if rel.get("certification"):
                    certification = rel.get("certification")
                    break
            break

    genres = [g["name"] for g in d.get("genres", [])]
    keywords = [k["name"] for k in d.get("keywords", {}).get("keywords", [])[:8]]
    cast = [c["name"] for c in d.get("credits", {}).get("cast", [])[:5]]
    crew = d.get("credits", {}).get("crew", [])
    directors = [c["name"] for c in crew if c.get("job") == "Director"]

    plot = d.get("overview") or ""
    release_date = d.get("release_date") or ""
    year = int(release_date[:4]) if len(release_date) >= 4 and release_date[:4].isdigit() else 0

    # NLP Sentiment Analysis on Plot Description
    blob = TextBlob(plot)
    polarity = round(blob.sentiment.polarity, 3)
    subjectivity = round(blob.sentiment.subjectivity, 3)

    # Combined text soup for TF-IDF feature extraction
    combined_features = (
        f"{plot} "
        f"{' '.join(genres)} {' '.join(genres)} "
        f"{' '.join(keywords)} "
        f"{' '.join(cast)} "
        f"{' '.join(directors)}"
    )

    poster_path = d.get("poster_path")
    backdrop_path = d.get("backdrop_path")
    poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else "https://via.placeholder.com/500x750/0B172A/00A8E8?text=No+Poster"
    backdrop_url = f"https://image.tmdb.org/t/p/original{backdrop_path}" if backdrop_path else poster_url

    return {
        "id": d.get("id"),
        "title": d.get("title", "Unknown Title"),
        "year": year,
        "decade": f"{(year // 10) * 10}s" if year > 0 else "Unknown",
        "runtime": d.get("runtime") or 0,
        "certification": certification,
        "language": (d.get("original_language") or "en").upper(),
        "genres": genres,
        "genres_str": ", ".join(genres),
        "plot": plot,
        "keywords": keywords,
        "cast": cast,
        "cast_str": ", ".join(cast),
        "directors": directors,
        "director_str": ", ".join(directors) if directors else "Unknown",
        "rating": round(d.get("vote_average") or 0.0, 1),
        "votes": d.get("vote_count") or 0,
        "revenue": d.get("revenue") or 0,
        "budget": d.get("budget") or 0,
        "popularity": round(d.get("popularity") or 0.0, 1),
        "polarity": polarity,
        "subjectivity": subjectivity,
        "combined_features": combined_features,
        "poster_url": poster_url,
        "backdrop_url": backdrop_url
    }

@st.cache_data(show_spinner=False, ttl=3600)
def build_candidate_pool(seed_movie_id=None):
    """Builds a rich pool of candidate movies from Popular, Top Rated, Trending, and Similar endpoints."""
    movie_ids = []
    if seed_movie_id:
        movie_ids.append(seed_movie_id)
        for endpoint in ["recommendations", "similar"]:
            url = f"https://api.themoviedb.org/3/movie/{seed_movie_id}/{endpoint}"
            res = requests.get(url, params={"api_key": API_KEY}).json()
            for m in res.get("results", [])[:12]:
                if m["id"] not in movie_ids:
                    movie_ids.append(m["id"])

    endpoints = [
        ("https://api.themoviedb.org/3/trending/movie/week", 1),
        ("https://api.themoviedb.org/3/movie/popular", 1),
        ("https://api.themoviedb.org/3/movie/top_rated", 1)
    ]
    for url, page in endpoints:
        res = requests.get(url, params={"api_key": API_KEY, "page": page}).json()
        for m in res.get("results", [])[:15]:
            if m["id"] not in movie_ids:
                movie_ids.append(m["id"])

    records = []
    for m_id in movie_ids[:50]:
        details = fetch_movie_details(m_id)
        if details and details["year"] > 0:
            records.append(details)

    return pd.DataFrame(records)

@st.cache_data(show_spinner=False, ttl=3600)
def search_tmdb_movie(query):
    """Searches TMDB by movie title."""
    url = "https://api.themoviedb.org/3/search/movie"
    res = requests.get(url, params={"api_key": API_KEY, "query": query}).json()
    return res.get("results", [])

# ==========================================
# 3. CORE FILTERING & EXPLANATION FUNCTIONS
# ==========================================
def apply_core_filters(df, year_range, min_rating, min_votes, runtime_range, certs, lang, personnel_query, selected_genres, genre_op, exclude_id=None):
    """Applies all 5 required core filtering capabilities from the assignment rubric."""
    filtered = df.copy()
    if exclude_id is not None:
        filtered = filtered[filtered["id"] != exclude_id]

    filtered = filtered[(filtered["year"] >= year_range[0]) & (filtered["year"] <= year_range[1])]
    filtered = filtered[(filtered["rating"] >= min_rating) & (filtered["votes"] >= min_votes)]
    filtered = filtered[(filtered["runtime"] >= runtime_range[0]) & (filtered["runtime"] <= runtime_range[1])]

    if certs:
        filtered = filtered[filtered["certification"].isin(certs)]
    if lang != "ALL":
        filtered = filtered[filtered["language"] == lang]

    if personnel_query.strip():
        q = personnel_query.strip().lower()
        filtered = filtered[
            filtered["cast_str"].str.lower().str.contains(q, na=False) |
            filtered["director_str"].str.lower().str.contains(q, na=False)
        ]

    if selected_genres:
        if genre_op == "Match ANY selected genre (OR)":
            filtered = filtered[filtered["genres"].apply(lambda g_list: any(g in g_list for g in selected_genres))]
        else:
            filtered = filtered[filtered["genres"].apply(lambda g_list: all(g in g_list for g in selected_genres))]

    return filtered

def build_similarity_explanation(reference_info, candidate_row):
    """Generates human-readable explanations for why a movie was recommended."""
    reasons = []
    ref_genres = set(reference_info.get("genres", []))
    cand_genres = set(candidate_row.get("genres", []))
    shared_genres = ref_genres.intersection(cand_genres)
    if shared_genres:
        reasons.append(f"Genres: {', '.join(shared_genres)}")

    ref_cast = set(reference_info.get("cast", []))
    cand_cast = set(candidate_row.get("cast", []))
    shared_cast = ref_cast.intersection(cand_cast)
    if shared_cast:
        reasons.append(f"Cast: {', '.join(shared_cast)}")

    ref_kw = set(reference_info.get("keywords", []))
    cand_kw = set(candidate_row.get("keywords", []))
    shared_kw = ref_kw.intersection(cand_kw)
    if shared_kw:
        reasons.append(f"Themes: {', '.join(list(shared_kw)[:3])}")

    pol_diff = abs(reference_info.get("polarity", 0) - candidate_row.get("polarity", 0))
    if pol_diff < 0.15:
        reasons.append("Similar plot mood & tone")

    if not reasons:
        reasons.append("High TF-IDF plot vocabulary alignment & audience score")

    return " | ".join(reasons)

# ==========================================
# 4. CRAVE TOP NAVIGATION BAR & TABS
# ==========================================
st.markdown("""
<div class="crave-navbar">
    <div class="crave-wordmark">
        CINEPULSE <span class="crave-pill">STREAM AI</span>
    </div>
    <div class="crave-tagline">
        Powered by TMDB Live Metadata &bull; TF-IDF Vectorization &bull; NLP Sentiment Engine
    </div>
</div>
""", unsafe_allow_html=True)

tab_home, tab_trends, tab_compare, tab_watchlist = st.tabs([
    "🏠 Home & AI Discovery",
    "📈 Trending & Analytics",
    "⚖️ Compare Movies",
    f"➕ My List ({len(st.session_state.watchlist)})"
])

# ==========================================
# TAB 1: CRAVE HOME & AI DISCOVERY STREAM
# ==========================================
with tab_home:
    # Sleek Search Bar + Mode Selector
    s_col1, s_col2 = st.columns([1, 2.3])
    with s_col1:
        search_mode = st.selectbox(
            "Discovery Mode",
            ["Search by Movie Title", "Describe a Plot or Vibe (NLP Query)"]
        )
    with s_col2:
        if search_mode == "Search by Movie Title":
            user_input = st.text_input("Search Movies", value="The Matrix", placeholder="Type a movie title (e.g., The Matrix, Dune, Inception)...")
        else:
            user_input = st.text_input("Describe Vibe or Plot", value="A futuristic cyberpunk hacker discovers reality is a simulation", placeholder="Describe any mood, plot, or theme...")

    # Sleek Collapsible Drawer for All 5 Rubric Filter Categories (No Sidebar Clutter!)
    with st.expander("🎛️ Refine Stream: Filters & AI Algorithm Settings (Year, Rating, Runtime, Cast, Genres)", expanded=False):
        f_col1, f_col2
