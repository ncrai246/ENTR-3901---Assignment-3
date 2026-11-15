import streamlit as st
import requests
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import plotly.express as px

try:
    from streamlit_searchbox import st_searchbox
    HAS_SEARCHBOX = True
except ImportError:
    HAS_SEARCHBOX = False

# ==========================================
# 1. PAGE CONFIGURATION & CRAVE STREAMING UI
# ==========================================
st.set_page_config(
    page_title="WatchThis! | Stream Smarter",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Crave Streaming Platform CSS Overhaul + Circular 'i' Hover Tooltips
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

/* Custom Filter Labels with Circular 'i' Hover Tooltips */
.filter-header {
    font-size: 1.02rem;
    font-weight: 800;
    color: #FFFFFF;
    margin-bottom: 12px;
    padding-bottom: 6px;
    border-bottom: 1px solid rgba(0, 168, 232, 0.25);
    display: flex;
    align-items: center;
    gap: 8px;
}
.filter-label {
    font-size: 0.86rem;
    font-weight: 600;
    color: #F0F6FC;
    margin-top: 6px;
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 7px;
}
.info-circle {
    position: relative;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 17px;
    height: 17px;
    border-radius: 50%;
    background: rgba(0, 168, 232, 0.18);
    border: 1.5px solid #00A8E8;
    color: #5CE1E6;
    font-family: Georgia, serif;
    font-size: 11px;
    font-weight: 700;
    font-style: italic;
    cursor: pointer;
    user-select: none;
    transition: all 0.2s ease;
}
.info-circle:hover {
    background: #00A8E8;
    color: #050A14;
    box-shadow: 0 0 10px rgba(0, 168, 232, 0.6);
}
.info-circle .tooltip-box {
    visibility: hidden;
    opacity: 0;
    width: 250px;
    background: #091528;
    color: #E2EEF9;
    text-align: left;
    font-family: 'Segoe UI', Roboto, Helvetica, sans-serif;
    font-size: 0.79rem;
    font-weight: 400;
    font-style: normal;
    line-height: 1.45;
    padding: 10px 12px;
    border-radius: 8px;
    border: 1px solid #00A8E8;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.85);
    position: absolute;
    z-index: 99999;
    bottom: 135%;
    left: 50%;
    transform: translateX(-50%);
    transition: opacity 0.18s ease;
    pointer-events: none;
}
.info-circle:hover .tooltip-box {
    visibility: visible;
    opacity: 1;
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
    margin-bottom: 14px;
    border: 1px solid rgba(0, 168, 232, 0.3);
    box-shadow: 0 16px 40px rgba(0, 0, 0, 0.65);
    background-size: cover;
    background-position: center 20%;
}
.hero-content {
    max-width: 760px;
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
    font-size: 0.98rem;
    line-height: 1.55;
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
    font-size: 0.98rem;
    font-weight: 700;
    color: #FFFFFF;
    line-height: 1.25;
    min-height: 2.5rem;
    margin-bottom: 4px;
}
.card-sub {
    font-size: 0.8rem;
    color: #8BA3C7;
    margin-bottom: 10px;
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
.detail-box {
    background: rgba(11, 23, 42, 0.9);
    border: 1px solid rgba(0, 168, 232, 0.3);
    border-radius: 12px;
    padding: 20px;
    margin-top: 12px;
    margin-bottom: 16px;
}
[data-testid="stMetricValue"] {
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: unset !important;
    font-size: 1.35rem !important;
    line-height: 1.3 !important;
}
</style>
""", unsafe_allow_html=True)

def render_label_with_info(label_text, tooltip_text, is_header=False):
    """Renders a clean label or section header with a circular 'i' hover tooltip."""
    css_class = "filter-header" if is_header else "filter-label"
    st.markdown(
        f'<div class="{css_class}">'
        f'<span>{label_text}</span>'
        f'<span class="info-circle" title="{tooltip_text}">i'
        f'<span class="tooltip-box">{tooltip_text}</span>'
        f'</span>'
        f'</div>',
        unsafe_allow_html=True
    )

# API Key setup using Streamlit Secrets
if "TMDB_KEY" in st.secrets:
    API_KEY = st.secrets["TMDB_KEY"]
else:
    st.error("API Key not found in Streamlit Secrets.")
    st.stop()

if "watchlist" not in st.session_state:
    st.session_state.watchlist = []
if "active_movie_id" not in st.session_state:
    st.session_state.active_movie_id = 603  # Defaults to The Matrix (1999)
if "detail_movie" not in st.session_state:
    st.session_state.detail_movie = None

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
    keywords = [k["name"] for k in d.get("keywords", {}).get("keywords", [])[:10]]
    cast = [c["name"] for c in d.get("credits", {}).get("cast", [])[:6]]
    crew = d.get("credits", {}).get("crew", [])
    directors = [c["name"] for c in crew if c.get("job") == "Director"]

    plot = d.get("overview") or "No plot synopsis available."
    release_date = d.get("release_date") or ""
    year = int(release_date[:4]) if len(release_date) >= 4 and release_date[:4].isdigit() else 0

    # NLP Sentiment Analysis on Plot Description
    try:
        blob = TextBlob(plot)
        polarity = round(blob.sentiment.polarity, 3)
        subjectivity = round(blob.sentiment.subjectivity, 3)
    except Exception:
        polarity = 0.0
        subjectivity = 0.5

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
    poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else "https://via.placeholder.com/500x750?text=No+Poster"
    backdrop_url = f"https://image.tmdb.org/t/p/original{backdrop_path}" if backdrop_path else poster_url
    title_clean = d.get("title", "Unknown Title").strip()

    return {
        "id": d.get("id"),
        "title": title_clean,
        "norm_title": title_clean.lower(),
        "year": year,
        "decade": f"{(year // 10) * 10}s" if year > 0 else "Unknown",
        "runtime": d.get("runtime") or 0,
        "certification": certification,
        "language": (d.get("original_language") or "en").upper(),
        "genres": genres,
        "genres_str": ", ".join(genres) if genres else "Unspecified",
        "plot": plot,
        "keywords": keywords,
        "keywords_str": ", ".join(keywords) if keywords else "None listed",
        "cast": cast,
        "cast_str": ", ".join(cast) if cast else "Unspecified",
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
    """Builds a rich, deduplicated pool of candidate movies from TMDB endpoints."""
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
    for m_id in movie_ids[:55]:
        details = fetch_movie_details(m_id)
        if details and details["year"] > 0:
            records.append(details)

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values(by=["popularity", "votes"], ascending=False)
        df = df.drop_duplicates(subset=["norm_title"], keep="first").reset_index(drop=True)
    return df

@st.cache_data(show_spinner=False, ttl=3600)
def search_tmdb_movie(query):
    """Searches TMDB by movie title and sorts by popularity to prioritize major releases."""
    url = "https://api.themoviedb.org/3/search/movie"
    res = requests.get(url, params={"api_key": API_KEY, "query": query}).json()
    results = res.get("results", [])
    results.sort(key=lambda x: (x.get("popularity") or 0, x.get("vote_count") or 0), reverse=True)
    return results

KNOWN_TITLES = [
    "The Matrix", "The Matrix Reloaded", "The Odyssey", "2001: A Space Odyssey",
    "Inception", "Interstellar", "Dune", "Dune: Part Two", "Blade Runner 2049",
    "Project Hail Mary", "The Dark Knight", "Spider-Man: Brand New Day",
    "Arrival", "Ex Machina", "Minority Report", "Avatar", "The Godfather",
    "Pulp Fiction", "Fight Club", "Forrest Gump", "Gladiator", "The Avengers",
    "Oppenheimer", "Barbie", "Parasite", "Whiplash", "Mad Max: Fury Road",
    "Alien", "Terminator 2: Judgment Day", "Back to the Future", "Jurassic Park"
]

def live_google_style_autocomplete(searchterm: str):
    """Callback for st_searchbox that cleans formatting, ranks by popularity, and removes duplicate titles."""
    if not searchterm:
        return []
    clean_q = searchterm.split("|")[0].split("(")[0].strip()
    if len(clean_q) < 2:
        return []

    options = []
    seen_title_years = set()

    results = search_tmdb_movie(clean_q)
    for m in results[:12]:
        m_id = m.get("id")
        title = (m.get("title") or "").strip()
        rel = (m.get("release_date") or "")[:4]
        year_str = rel if rel else "TBA"
        rating = round(m.get("vote_average") or 0.0, 1)
        dedup_key = (title.lower(), year_str)
        if m_id and title and dedup_key not in seen_title_years:
            label = f"{title} ({year_str}) | ★ {rating}"
            options.append((label, m_id))
            seen_title_years.add(dedup_key)

    if len(options) < 3:
        close_matches = difflib.get_close_matches(clean_q, KNOWN_TITLES, n=3, cutoff=0.45)
        for match_title in close_matches:
            fuzzy_res = search_tmdb_movie(match_title)
            for m in fuzzy_res[:4]:
                m_id = m.get("id")
                title = (m.get("title") or "").strip()
                rel = (m.get("release_date") or "")[:4]
                year_str = rel if rel else "TBA"
                rating = round(m.get("vote_average") or 0.0, 1)
                dedup_key = (title.lower(), year_str)
                if m_id and title and dedup_key not in seen_title_years:
                    label = f"{title} ({year_str}) | ★ {rating} (Suggested)"
                    options.append((label, m_id))
                    seen_title_years.add(dedup_key)

    return options[:8]

# ==========================================
# 3. CORE FILTERING & EXPLANATION FUNCTIONS
# ==========================================
def apply_core_filters(df, year_range, min_rating, min_votes, runtime_range, certs, lang, personnel_query, selected_genres, genre_op, exclude_id=None, exclude_title=None):
    """Applies all 5 required core filtering capabilities and removes same-title duplicates."""
    filtered = df.copy()
    if exclude_id is not None:
        filtered = filtered[filtered["id"] != exclude_id]
    if exclude_title:
        filtered = filtered[filtered["norm_title"] != exclude_title.strip().lower()]

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
        reasons.append(f"Shared Genres: {', '.join(shared_genres)}")

    ref_cast = set(reference_info.get("cast", []))
    cand_cast = set(candidate_row.get("cast", []))
    shared_cast = ref_cast.intersection(cand_cast)
    if shared_cast:
        reasons.append(f"Shared Cast: {', '.join(shared_cast)}")

    ref_kw = set(reference_info.get("keywords", []))
    cand_kw = set(candidate_row.get("keywords", []))
    shared_kw = ref_kw.intersection(cand_kw)
    if shared_kw:
        reasons.append(f"Matching Themes: {', '.join(list(shared_kw)[:4])}")

    pol_diff = abs(reference_info.get("polarity", 0) - candidate_row.get("polarity", 0))
    if pol_diff < 0.15:
        reasons.append("Highly similar narrative mood & emotional tone")

    if not reasons:
        reasons.append("Strong TF-IDF plot vocabulary alignment & high audience quality score")

    return " | ".join(reasons)

def render_full_movie_page(m_data):
    """Renders a dedicated Full-Page Movie Detail View when any movie title is clicked."""
    top_b1, top_b2, top_b3 = st.columns([1.2, 1.5, 1.3])
    with top_b1:
        if st.button("⬅️ Back to Discovery Stream", type="primary", use_container_width=True):
            st.session_state.detail_movie = None
            st.rerun()
    with top_b2:
        if st.button("🔍 Find Movies Similar to This Title", use_container_width=True):
            st.session_state.active_movie_id = m_data["id"]
            st.session_state.detail_movie = None
            st.rerun()
    with top_b3:
        if st.button("➕ Add Title to My List", use_container_width=True):
            if m_data["id"] not in [w["id"] for w in st.session_state.watchlist]:
                st.session_state.watchlist.append(m_data)
                st.toast(f"Added {m_data['title']} to My List!")

    st.markdown(f"""
    <div class="hero-banner" style="background-image: linear-gradient(90deg, #050A14 12%, rgba(5, 10, 20, 0.85) 55%, rgba(5, 10, 20, 0.3) 100%), url('{m_data['backdrop_url']}');">
        <div class="hero-content">
            <div class="hero-eyebrow">FULL MOVIE SPOTLIGHT &bull; COMPLETE METADATA VIEW</div>
            <div class="hero-title">{m_data['title']} ({m_data['year']})</div>
            <div class="hero-meta">
                <span class="meta-badge">★ {m_data['rating']}/10 TMDB ({m_data['votes']:,} votes)</span>
                <span class="meta-badge">{m_data['certification']}</span>
                <span class="meta-badge">{m_data['language']}</span>
                <span>{m_data['runtime']} MINS</span>
                <span>&bull;</span>
                <span>{m_data['genres_str']}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    d_col1, d_col2 = st.columns([1, 2.6])
    with d_col1:
        st.image(m_data["poster_url"], use_container_width=True)
    with d_col2:
        st.markdown("### 📖 Complete Plot Synopsis")
        st.write(m_data["plot"])

        if "explanation" in m_data:
            st.markdown(f"""
            <div class="detail-box">
                <div style="color:#00A8E8; font-weight:800; font-size:0.9rem; letter-spacing:1px; margin-bottom:6px;">
                    🤖 AI RECOMMENDATION BREAKDOWN ({m_data.get('match_pct', 85)}% STREAM MATCH)
                </div>
                <div style="font-size:0.98rem; color:#F0F6FC; margin-bottom:8px;">
                    <b>Why It Matches Your Search:</b> {m_data['explanation']}
                </div>
                <div style="font-size:0.88rem; color:#8AB4F8;">
                    Content TF-IDF Score: <code>{m_data.get('content_score', 0):.2f}</code> &nbsp;|&nbsp;
                    NLP Sentiment Alignment: <code>{m_data.get('sentiment_score', 0):.2f}</code> &nbsp;|&nbsp;
                    Plot Polarity: <code>{m_data.get('polarity', 0)}</code> &nbsp;|&nbsp;
                    Plot Subjectivity: <code>{m_data.get('subjectivity', 0)}</code>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 🎬 Production, Cast & Box Office Metadata")
        rev_str = f"${m_data['revenue']:,}" if m_data["revenue"] > 0 else "Unreported"
        bud_str = f"${m_data['budget']:,}" if m_data["budget"] > 0 else "Unreported"
        m1, m2, m3 = st.columns(3)
        m1.markdown(
            f'<div class="detail-box" style="margin-top:0;">'
            f'<div style="color:#8BA3C7; font-size:0.85rem;">Director</div>'
            f'<div style="color:#FFFFFF; font-size:1.2rem; font-weight:700; line-height:1.35; margin-top:4px;">{m_data["director_str"]}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
        m2.markdown(
            f'<div class="detail-box" style="margin-top:0;">'
            f'<div style="color:#8BA3C7; font-size:0.85rem;">Box Office Revenue</div>'
            f'<div style="color:#FFFFFF; font-size:1.2rem; font-weight:700; line-height:1.35; margin-top:4px;">{rev_str}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
        m3.markdown(
            f'<div class="detail-box" style="margin-top:0;">'
            f'<div style="color:#8BA3C7; font-size:0.85rem;">Production Budget</div>'
            f'<div style="color:#FFFFFF; font-size:1.2rem; font-weight:700; line-height:1.35; margin-top:4px;">{bud_str}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown(f"**🌟 Top Billed Cast:** {m_data['cast_str']}")
        st.markdown(f"**🎭 Genres:** {m_data['genres_str']}")
        st.markdown(f"**🏷️ TMDB Plot Keywords & Themes:** {m_data.get('keywords_str', 'None listed')}")
        st.markdown(f"**⏱️ Runtime:** {m_data['runtime']} minutes &nbsp;|&nbsp; **🔞 US Certification:** {m_data['certification']} &nbsp;|&nbsp; **🌐 Original Language:** {m_data['language']}")

# ==========================================
# 4. CRAVE TOP NAVIGATION BAR & TABS
# ==========================================
st.markdown("""
<div class="crave-navbar">
    <div class="crave-wordmark">
        WatchThis! <span class="crave-pill">STREAM AI</span>
    </div>
    <div class="crave-tagline">
        Powered by TMDB Live Metadata &bull; TF-IDF Vectorization &bull; NLP Sentiment Engine
    </div>
</div>
""", unsafe_allow_html=True)

df_base_catalog = build_candidate_pool(seed_movie_id=None)

# Global Full-Page Movie Spotlight Check (Works Immediately From Any Tab!)
if st.session_state.detail_movie is not None:
    render_full_movie_page(st.session_state.detail_movie)
else:
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
        s_col1, s_col2 = st.columns([1, 2.5])
        with s_col1:
            render_label_with_info(
                "Discovery Mode",
                "Choose whether to search for a specific movie title you already know, or type a sentence describing the plot, mood, or vibe you feel like watching."
            )
            search_mode = st.selectbox(
                "Discovery Mode",
                ["Search by Movie Title", "Describe a Plot or Vibe (NLP Query)"],
                label_visibility="collapsed"
            )

        user_nlp_query = ""
        with s_col2:
            if search_mode == "Search by Movie Title":
                render_label_with_info(
                    "Search Movies (Live Auto-Complete Dropdown)",
                    "Start typing any movie title. A live dropdown list will automatically appear below the bar as you type, even if you make a spelling mistake."
                )
                if HAS_SEARCHBOX:
                    picked_result = st_searchbox(
                        live_google_style_autocomplete,
                        placeholder="Start typing any movie title (e.g., The Matrix, The Odyssey, Dune)...",
                        key="crave_live_searchbox"
                    )
                    if picked_result is not None:
                        if isinstance(picked_result, tuple):
                            st.session_state.active_movie_id = picked_result[1]
                        elif isinstance(picked_result, int):
                            st.session_state.active_movie_id = picked_result
                else:
                    fallback_q = st.text_input("Search Movies", value="The Matrix", label_visibility="collapsed")
                    res = search_tmdb_movie(fallback_q)
                    if res:
                        st.session_state.active_movie_id = res[0]["id"]
            else:
                render_label_with_info(
                    "Describe Vibe, Plot, or Themes",
                    "Type any plain-English description of what you want to watch (like 'a mind-bending space adventure' or 'an uplifting underdog sports story')."
                )
                user_nlp_query = st.text_input(
                    "Describe Vibe, Plot, or Themes",
                    value="A futuristic cyberpunk hacker discovers reality is a simulation",
                    placeholder="Describe any mood, plot, or theme...",
                    label_visibility="collapsed"
                )

        # Renamed Drawer with Circular 'i' Plain-English Tooltips on Every Control
        with st.expander("🎛️ Advanced Filtering", expanded=False):
            f_col1, f_col2, f_col3 = st.columns(3)
            with f_col1:
                render_label_with_info(
                    "AI Engine & Temporal Filter",
                    "Control how the AI brain ranks recommendations and narrow down movies by when they came out or how long they run.",
                    is_header=True
                )
                render_label_with_info(
                    "Recommendation Algorithm",
                    "Picks how the AI finds matches: Hybrid blends plot, mood, and viewer ratings; Content-Based matches storyline keywords, cast, and genres; Sentiment-Based matches the emotional tone of the story."
                )
                rec_approach = st.selectbox(
                    "Recommendation Algorithm",
                    [
                        "Hybrid (Content + NLP Sentiment + Quality)",
                        "Content-Based Filtering (TF-IDF Attributes)",
                        "Sentiment-Based NLP Matching (Plot Tone)"
                    ],
                    label_visibility="collapsed"
                )
                render_label_with_info(
                    "Release Year Range",
                    "Only show movies released between these two years. Slide the handles to focus on modern releases or classic eras."
                )
                year_range = st.slider("Release Year Range", 1970, 2026, (1990, 2026), label_visibility="collapsed")
                render_label_with_info(
                    "Runtime Bracket (mins)",
                    "Filter movies by their total length in minutes. Great when you want a quick 90-minute movie night or a 3-hour epic."
                )
                runtime_range = st.slider("Runtime Bracket (mins)", 40, 240, (60, 220), label_visibility="collapsed")

            with f_col2:
                render_label_with_info(
                    "Quality & Content Specifications",
                    "Set quality standards for viewer ratings and filter by age appropriateness or spoken language.",
                    is_header=True
                )
                render_label_with_info(
                    "Minimum TMDB Rating",
                    "Hides poorly rated movies. Only movies with an average fan score equal to or higher than this number (out of 10) will be shown."
                )
                min_rating = st.slider("Minimum TMDB Rating", 0.0, 10.0, 6.0, 0.5, label_visibility="collapsed")
                render_label_with_info(
                    "Minimum Vote Count",
                    "Makes sure ratings are trustworthy by requiring a minimum number of viewer reviews. Raising this hides obscure movies with only a few votes."
                )
                min_votes = st.number_input("Minimum Vote Count", min_value=0, max_value=10000, value=50, step=50, label_visibility="collapsed")
                render_label_with_info(
                    "US Certification",
                    "Filter by official MPAA age rating: G (General Audiences), PG (Parental Guidance), PG-13 (Parents Strongly Cautioned), R (Restricted 17+), or NR (Not Rated)."
                )
                all_certs = ["G", "PG", "PG-13", "R", "NR"]
                selected_certs = st.multiselect("US Certification", all_certs, default=all_certs, label_visibility="collapsed")
                render_label_with_info(
                    "Original Language",
                    "Filter movies by the primary language they were originally filmed in (All Languages, English, French, Spanish, Japanese, or Korean)."
                )
                selected_lang = st.selectbox("Original Language", ["ALL", "EN", "FR", "ES", "JA", "KO"], label_visibility="collapsed")

            with f_col3:
                render_label_with_info(
                    "Personnel & Genre Combinations",
                    "Search for your favorite actors or directors and combine movie genres.",
                    is_header=True
                )
                render_label_with_info(
                    "Actor or Director Preference",
                    "Type the name of an actor or director (such as Keanu Reeves or Christopher Nolan) to only show movies they worked on."
                )
                personnel_query = st.text_input("Actor or Director Preference", placeholder="e.g., Keanu Reeves, Christopher Nolan", label_visibility="collapsed")
                render_label_with_info(
                    "Filter by Genres",
                    "Select one or more movie categories (such as Action, Sci-Fi, or Comedy) to narrow down your recommendations."
                )
                genre_options = ["Action", "Adventure", "Animation", "Comedy", "Crime", "Drama", "Family", "Fantasy", "Horror", "Mystery", "Romance", "Science Fiction", "Thriller"]
                selected_genres = st.multiselect("Filter by Genres", genre_options, label_visibility="collapsed")
                render_label_with_info(
                    "Genre Logic",
                    "Controls how multiple selected genres work together: 'ANY (OR)' shows movies that have at least one of your chosen genres; 'ALL (AND)' only shows movies that contain every single genre you picked."
                )
                genre_operator = st.radio("Genre Logic", ["Match ANY selected genre (OR)", "Match ALL selected genres (AND)"], label_visibility="collapsed")

        with st.spinner("Curating your personalized stream..."):
            seed_movie = None
            if search_mode == "Search by Movie Title":
                seed_movie = fetch_movie_details(st.session_state.active_movie_id)
                df_pool = build_candidate_pool(seed_movie_id=st.session_state.active_movie_id)
            else:
                df_pool = df_base_catalog.copy()

            vectorizer = TfidfVectorizer(stop_words="english")
            tfidf_matrix = vectorizer.fit_transform(df_pool["combined_features"])

            if seed_movie is not None:
                target_vec = vectorizer.transform([seed_movie["combined_features"]])
                ref_polarity = seed_movie["polarity"]
                ref_subjectivity = seed_movie["subjectivity"]
                ref_info = seed_movie
                exclude_id = seed_movie["id"]
                exclude_title = seed_movie["title"]
                hero_movie = seed_movie
                eyebrow_text = f"FEATURED SELECTION &bull; SELECTED TITLE: {seed_movie['title'].upper()}"
            else:
                try:
                    q_blob = TextBlob(user_nlp_query)
                    ref_polarity = round(q_blob.sentiment.polarity, 3)
                    ref_subjectivity = round(q_blob.sentiment.subjectivity, 3)
                except Exception:
                    ref_polarity = 0.0
                    ref_subjectivity = 0.5
                target_vec = vectorizer.transform([user_nlp_query])
                ref_info = {"genres": selected_genres, "cast": [], "keywords": user_nlp_query.lower().split(), "polarity": ref_polarity}
                exclude_id = None
                exclude_title = None
                hero_movie = df_pool.iloc[0].to_dict()
                eyebrow_text = f"NLP VIBE MATCH &bull; DETECTED POLARITY: {ref_polarity}"

            df_pool["content_score"] = cosine_similarity(target_vec, tfidf_matrix).flatten()
            pol_diff = (df_pool["polarity"] - ref_polarity).abs()
            sub_diff = (df_pool["subjectivity"] - ref_subjectivity).abs()
            df_pool["sentiment_score"] = (1.0 - ((pol_diff + sub_diff) / 3.0)).clip(0, 1)

            if rec_approach.startswith("Content-Based"):
                df_pool["final_score"] = df_pool["content_score"]
            elif rec_approach.startswith("Sentiment-Based"):
                df_pool["final_score"] = (0.75 * df_pool["sentiment_score"]) + (0.25 * df_pool["content_score"])
            else:
                norm_rating = df_pool["rating"] / 10.0
                df_pool["final_score"] = (0.55 * df_pool["content_score"]) + (0.25 * df_pool["sentiment_score"]) + (0.20 * norm_rating)

            filtered_df = apply_core_filters(
                df_pool, year_range, min_rating, min_votes, runtime_range,
                selected_certs, selected_lang, personnel_query, selected_genres,
                genre_operator, exclude_id=exclude_id, exclude_title=exclude_title
            ).sort_values(by="final_score", ascending=False)

            st.markdown(f"""
            <div class="hero-banner" style="background-image: linear-gradient(90deg, #050A14 15%, rgba(5, 10, 20, 0.82) 55%, rgba(5, 10, 20, 0.25) 100%), url('{hero_movie['backdrop_url']}');">
                <div class="hero-content">
                    <div class="hero-eyebrow">{eyebrow_text}</div>
                    <div class="hero-title">{hero_movie['title']} ({hero_movie['year']})</div>
                    <div class="hero-meta">
                        <span class="meta-badge">★ {hero_movie['rating']} TMDB</span>
                        <span class="meta-badge">{hero_movie['certification']}</span>
                        <span>{hero_movie['runtime']} MINS</span>
                        <span>&bull;</span>
                        <span>{hero_movie['genres_str']}</span>
                        <span>&bull;</span>
                        <span>Dir: {hero_movie['director_str']}</span>
                    </div>
                    <div class="hero-plot">{hero_movie['plot']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Action Buttons for the Searched / Featured Hero Movie
            h_col1, h_col2, _ = st.columns([1.3, 1.3, 3.4])
            with h_col1:
                if st.button("🎬 Open Full Page", key=f"hero_open_{hero_movie['id']}", type="primary", use_container_width=True):
                    st.session_state.detail_movie = hero_movie
                    st.rerun()
            with h_col2:
                if st.button("➕ Add to My List", key=f"hero_add_{hero_movie['id']}", use_container_width=True):
                    if hero_movie["id"] not in [w["id"] for w in st.session_state.watchlist]:
                        st.session_state.watchlist.append(hero_movie)
                        st.toast(f"Added {hero_movie['title']} to My List!")
                        st.rerun()

            st.markdown('<div class="section-header">Recommended For You (Click Any Movie to Open Full Page)</div>', unsafe_allow_html=True)
            if filtered_df.empty:
                st.warning("No titles matched all of your active filter settings. Open 'Advanced Filtering' above to widen your year, genre, or rating range.")
            else:
                rec_cols = st.columns(5)
                for idx, (_, row) in enumerate(filtered_df.head(10).iterrows()):
                    with rec_cols[idx % 5]:
                        with st.container(border=True):
                            st.image(row["poster_url"], use_container_width=True)
                            match_pct = int(min(max(row["final_score"] * 100 + 35, 65), 99))
                            st.markdown(f'<div class="match-pill">{match_pct}% STREAM MATCH</div>', unsafe_allow_html=True)
                            st.markdown(f'<div class="card-title">{row["title"]}</div>', unsafe_allow_html=True)
                            st.markdown(f'<div class="card-sub">{row["year"]} &bull; ★ {row["rating"]} &bull; {row["certification"]}</div>', unsafe_allow_html=True)

                            explanation = build_similarity_explanation(ref_info, row)
                            if st.button("🎬 Open Full Page", key=f"open_{row['id']}", type="primary", use_container_width=True):
                                m_dict = row.to_dict()
                                m_dict["explanation"] = explanation
                                m_dict["match_pct"] = match_pct
                                st.session_state.detail_movie = m_dict
                                st.rerun()

                            if st.button("➕ My List", key=f"rec_{row['id']}", use_container_width=True):
                                if row["id"] not in [w["id"] for w in st.session_state.watchlist]:
                                    st.session_state.watchlist.append(row.to_dict())
                                    st.toast(f"Added {row['title']} to My List!")
                                    st.rerun()

    # ==========================================
    # TAB 2: TRENDING & MARKET ANALYTICS
    # ==========================================
    with tab_trends:
        st.markdown('<div class="section-header">Trending Now on WatchThis!</div>', unsafe_allow_html=True)
        df_trends = df_base_catalog.copy()

        t_cols = st.columns(5)
        for idx, (_, m_row) in enumerate(df_trends.head(5).iterrows()):
            with t_cols[idx % 5]:
                with st.container(border=True):
                    st.image(m_row["poster_url"], use_container_width=True)
                    st.markdown(f'<div class="match-pill">🔥 TRENDING #{idx + 1}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="card-title">{m_row["title"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="card-sub">{m_row["year"]} &bull; ★ {m_row["rating"]} &bull; {m_row["certification"]}</div>', unsafe_allow_html=True)
                    if st.button("🎬 Open Full Page", key=f"tropen_{m_row['id']}", type="primary", use_container_width=True):
                        st.session_state.detail_movie = m_row.to_dict()
                        st.rerun()
                    if st.button("➕ My List", key=f"tr_{m_row['id']}", use_container_width=True):
                        if m_row["id"] not in [w["id"] for w in st.session_state.watchlist]:
                            st.session_state.watchlist.append(m_row.to_dict())
                            st.toast(f"Added {m_row['title']} to My List!")
                            st.rerun()

        st.markdown('<div class="section-header">Interactive Catalog Visualizations</div>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        crave_palette = ["#00A8E8", "#38BDF8", "#0072CE", "#7DD3FC", "#0284C7", "#BAE6FD"]

        with col_a:
            fig_scatter = px.scatter(
                df_trends,
                x="rating",
                y="popularity",
                size="votes",
                color="decade",
                hover_name="title",
                hover_data=["year", "genres_str", "revenue"],
                title="Audience Rating vs. Popularity Score (Sized by Vote Count)",
                template="plotly_dark",
                color_discrete_sequence=crave_palette
            )
            fig_scatter.update_layout(
                paper_bgcolor="rgba(11, 23, 42, 0.85)",
                plot_bgcolor="rgba(5, 10, 20, 0.85)"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col_b:
            fig_sent = px.scatter(
                df_trends,
                x="polarity",
                y="subjectivity",
                color="rating",
                hover_name="title",
                title="NLP Plot Sentiment Map: Polarity vs. Subjectivity",
                template="plotly_dark",
                color_continuous_scale=["#09162E", "#0072CE", "#00A8E8", "#7DD3FC"]
            )
            fig_sent.update_layout(
                paper_bgcolor="rgba(11, 23, 42, 0.85)",
                plot_bgcolor="rgba(5, 10, 20, 0.85)"
            )
            st.plotly_chart(fig_sent, use_container_width=True)

    # ==========================================
    # TAB 3: COMPARATIVE MOVIE ANALYSIS
    # ==========================================
    with tab_compare:
        st.markdown('<div class="section-header">Side-by-Side Title Comparison</div>', unsafe_allow_html=True)
        df_comp = df_base_catalog.copy()
        movie_titles = df_comp["title"].tolist()

        c1, c2 = st.columns(2)
        with c1:
            movie_a_title = st.selectbox("Select First Title", movie_titles, index=0)
        with c2:
            default_b = 1 if len(movie_titles) > 1 else 0
            movie_b_title = st.selectbox("Select Second Title", movie_titles, index=default_b)

        row_a = df_comp[df_comp["title"] == movie_a_title].iloc[0]
        row_b = df_comp[df_comp["title"] == movie_b_title].iloc[0]

        p_col1, p_col2 = st.columns(2)
        with p_col1:
            with st.container(border=True):
                ic1, ic2 = st.columns([1, 2])
                with ic1:
                    st.image(row_a["poster_url"], use_container_width=True)
                with ic2:
                    st.subheader(f"{row_a['title']} ({row_a['year']})")
                    st.caption(f"★ {row_a['rating']}/10 | {row_a['runtime']} mins | {row_a['certification']}")
                    st.write(f"**Genres:** {row_a['genres_str']}")
                    st.write(f"**Director:** {row_a['director_str']}")
                    st.write(f"**Box Office:** ${row_a['revenue']:,}")
                    st.write(f"**NLP Sentiment Polarity:** `{row_a['polarity']}`")

        with p_col2:
            with st.container(border=True):
                ic3, ic4 = st.columns([1, 2])
                with ic3:
                    st.image(row_b["poster_url"], use_container_width=True)
                with ic4:
                    st.subheader(f"{row_b['title']} ({row_b['year']})")
                    st.caption(f"★ {row_b['rating']}/10 | {row_b['runtime']} mins | {row_b['certification']}")
                    st.write(f"**Genres:** {row_b['genres_str']}")
                    st.write(f"**Director:** {row_b['director_str']}")
                    st.write(f"**Box Office:** ${row_b['revenue']:,}")
                    st.write(f"**NLP Sentiment Polarity:** `{row_b['polarity']}`")

        comp_table = pd.DataFrame({
            "Attribute": ["Release Year", "TMDB Rating", "Vote Count", "Runtime (mins)", "US Certification", "Box Office Revenue", "Plot Sentiment Polarity", "Plot Subjectivity"],
            row_a["title"]: [str(row_a["year"]), f"{row_a['rating']}/10", f"{row_a['votes']:,}", str(row_a["runtime"]), row_a["certification"], f"${row_a['revenue']:,}", str(row_a["polarity"]), str(row_a["subjectivity"])],
            row_b["title"]: [str(row_b["year"]), f"{row_b['rating']}/10", f"{row_b['votes']:,}", str(row_b["runtime"]), row_b["certification"], f"${row_b['revenue']:,}", str(row_b["polarity"]), str(row_b["subjectivity"])]
        })
        st.dataframe(comp_table, use_container_width=True, hide_index=True)

    # ==========================================
    # TAB 4: USER WATCHLIST & PROFILE RECOMMENDATIONS
    # ==========================================
    with tab_watchlist:
        st.markdown('<div class="section-header">My Saved List & Profile Recommendations</div>', unsafe_allow_html=True)
        w_col1, w_col2 = st.columns([3, 1])
        with w_col1:
            add_title = st.text_input("Quickly add any movie by title to My List:", placeholder="e.g., Interstellar, Blade Runner 2049")
        with w_col2:
            st.write("")
            st.write("")
            if st.button("➕ Search & Add", use_container_width=True) and add_title.strip():
                res = search_tmdb_movie(add_title.strip())
                if res:
                    details = fetch_movie_details(res[0]["id"])
                    if details and details["id"] not in [w["id"] for w in st.session_state.watchlist]:
                        st.session_state.watchlist.append(details)
                        st.success(f"Added {details['title']} to My List!")
                        st.rerun()

        if not st.session_state.watchlist:
            st.info("Your list is currently empty. Click '➕ My List' on any movie card on the Home or Trending tabs to build your personal taste profile.")
        else:
            wl_cols = st.columns(5)
            for idx, item in enumerate(st.session_state.watchlist):
                with wl_cols[idx % 5]:
                    with st.container(border=True):
                        st.image(item["poster_url"], use_container_width=True)
                        st.markdown(f'<div class="card-title">{item["title"]}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="card-sub">{item["year"]} &bull; ★ {item["rating"]}</div>', unsafe_allow_html=True)
                        if st.button("🎬 Open Full Page", key=f"wlopen_{item['id']}", type="primary", use_container_width=True):
                            st.session_state.detail_movie = item
                            st.rerun()

            if st.button("🗑️ Clear My List"):
                st.session_state.watchlist = []
                st.rerun()

            st.markdown('<div class="section-header">Because You Saved These Titles</div>', unsafe_allow_html=True)
            df_all = df_base_catalog.copy()
            watchlist_ids = [w["id"] for w in st.session_state.watchlist]
            combined_profile_text = " ".join([w["combined_features"] for w in st.session_state.watchlist])
            vec = TfidfVectorizer(stop_words="english")
            mat = vec.fit_transform(df_all["combined_features"])
            prof_vec = vec.transform([combined_profile_text])
            df_all["profile_score"] = cosine_similarity(prof_vec, mat).flatten()
            recs = df_all[~df_all["id"].isin(watchlist_ids)].sort_values(by="profile_score", ascending=False).head(5)

            p_cols = st.columns(5)
            for idx, (_, r) in enumerate(recs.iterrows()):
                with p_cols[idx % 5]:
                    with st.container(border=True):
                        st.image(r["poster_url"], use_container_width=True)
                        pct = int(min(max(r["profile_score"] * 100 + 40, 70), 99))
                        st.markdown(f'<div class="match-pill">{pct}% PROFILE MATCH</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="card-title">{r["title"]}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="card-sub">{r["year"]} &bull; ★ {r["rating"]}</div>', unsafe_allow_html=True)
                        if st.button("🎬 Open Full Page", key=f"profopen_{r['id']}", type="primary", use_container_width=True):
                            st.session_state.detail_movie = r.to_dict()
                            st.rerun()

# END OF FILE
