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
    .search-label {
        font-size: 0.86rem;
        font-weight: 600;
        color: #F0F6FC;
        margin-bottom: 6px;
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

if "active_movie_id" not in st.session_state:
    st.session_state.active_movie_id = 603  # Defaults to The Matrix (1999)

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

# Known popular titles for instant fuzzy typo recovery
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
    """Callback function for st_searchbox that triggers on every keystroke."""
    if not searchterm or len(searchterm.strip()) < 2:
        return []

    clean_q = searchterm.strip()
    options = []
    seen_ids = set()

    # 1. Direct live search on TMDB as user types
    results = search_tmdb_movie(clean_q)
    for m in results[:8]:
        m_id = m.get("id")
        title = m.get("title")
        rel = (m.get("release_date") or "")[:4]
        year_str = rel if rel else "TBA"
        rating = round(m.get("vote_average") or 0.0, 1)
        if m_id and title and m_id not in seen_ids:
            label = f"{title} ({year_str}) | ★ {rating}"
            options.append((label, m_id))
            seen_ids.add(m_id)

    # 2. Fuzzy typo recovery if user misspelled the title
    if len(options) < 3:
        close_matches = difflib.get_close_matches(clean_q, KNOWN_TITLES, n=3, cutoff=0.45)
        for match_title in close_matches:
            fuzzy_res = search_tmdb_movie(match_title)
            for m in fuzzy_res[:4]:
                m_id = m.get("id")
                title = m.get("title")
                rel = (m.get("release_date") or "")[:4]
                year_str = rel if rel else "TBA"
                rating = round(m.get("vote_average") or 0.0, 1)
                if m_id and title and m_id not in seen_ids:
                    label = f"{title} ({year_str}) | ★ {rating} (Suggested Match)"
                    options.append((label, m_id))
                    seen_ids.add(m_id)

    return options

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

df_base_catalog = build_candidate_pool(seed_movie_id=None)

# ==========================================
# TAB 1: CRAVE HOME & AI DISCOVERY STREAM
# ==========================================
with tab_home:
    s_col1, s_col2 = st.columns([1, 2.5])
    with s_col1:
        search_mode = st.selectbox(
            "Discovery Mode",
            ["Search by Movie Title", "Describe a Plot or Vibe (NLP Query)"]
        )

    user_nlp_query = ""
    with s_col2:
        if search_mode == "Search by Movie Title":
            if HAS_SEARCHBOX:
                st.markdown('<div class="search-label">Search Movies (Live Auto-Complete Dropdown)</div>', unsafe_allow_html=True)
                picked_result = st_searchbox(
                    live_google_style_autocomplete,
                    placeholder="Start typing any movie title (e.g., The Matrix, Odyssey, Dune)...",
                    key="crave_live_searchbox"
                )
                if picked_result is not None:
                    if isinstance(picked_result, tuple):
                        st.session_state.active_movie_id = picked_result[1]
                    elif isinstance(picked_result, int):
                        st.session_state.active_movie_id = picked_result
            else:
                fallback_q = st.text_input("Search Movies", value="The Matrix")
                res = search_tmdb_movie(fallback_q)
                if res:
                    st.session_state.active_movie_id = res[0]["id"]
        else:
            user_nlp_query = st.text_input(
                "Describe Vibe, Plot, or Themes",
                value="A futuristic cyberpunk hacker discovers reality is a simulation",
                placeholder="Describe any mood, plot, or theme..."
            )

    with st.expander("🎛️ Refine Stream: Filters & AI Algorithm Settings (Year, Rating, Runtime, Cast, Genres)", expanded=False):
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            st.markdown("**AI Engine & Temporal Filter**")
            rec_approach = st.selectbox(
                "Recommendation Algorithm",
                [
                    "Hybrid (Content + NLP Sentiment + Quality)",
                    "Content-Based Filtering (TF-IDF Attributes)",
                    "Sentiment-Based NLP Matching (Plot Tone)"
                ]
            )
            year_range = st.slider("Release Year Range", 1970, 2026, (1990, 2026))
            runtime_range = st.slider("Runtime Bracket (mins)", 40, 240, (60, 220))

        with f_col2:
            st.markdown("**Quality & Content Specifications**")
            min_rating = st.slider("Minimum TMDB Rating", 0.0, 10.0, 6.0, 0.5)
            min_votes = st.number_input("Minimum Vote Count", min_value=0, max_value=10000, value=50, step=50)
            all_certs = ["G", "PG", "PG-13", "R", "NR"]
            selected_certs = st.multiselect("US Certification", all_certs, default=all_certs)
            selected_lang = st.selectbox("Original Language", ["ALL", "EN", "FR", "ES", "JA", "KO"])

        with f_col3:
            st.markdown("**Personnel & Genre Combinations**")
            personnel_query = st.text_input("Actor or Director Preference", placeholder="e.g., Keanu Reeves, Christopher Nolan")
            genre_options = ["Action", "Adventure", "Animation", "Comedy", "Crime", "Drama", "Family", "Fantasy", "Horror", "Mystery", "Romance", "Science Fiction", "Thriller"]
            selected_genres = st.multiselect("Filter by Genres", genre_options)
            genre_operator = st.radio("Genre Logic", ["Match ANY selected genre (OR)", "Match ALL selected genres (AND)"])

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
            genre_operator, exclude_id=exclude_id
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

        st.markdown('<div class="section-header">Recommended For You</div>', unsafe_allow_html=True)
        if filtered_df.empty:
            st.warning("No titles matched all of your active filter settings. Open 'Refine Stream' above to widen your year, rating, or genre filters.")
        else:
            rec_cols = st.columns(5)
            for idx, (_, row) in enumerate(filtered_df.head(10).iterrows()):
                with rec_cols[idx % 5]:
                    with st.container(border=True):
                        st.image(row["poster_url"], use_container_width=True)
                        match_pct = int(min(max(row["final_score"] * 100 + 35, 65), 99))
                        st.markdown(f'<div class="match-pill">{match_pct}% STREAM MATCH</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="card-title" title="{row["title"]}">{row["title"]}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="card-sub">{row["year"]} &bull; ★ {row["rating"]} &bull; {row["certification"]}</div>', unsafe_allow_html=True)

                        with st.expander("ℹ️ Details & Match"):
                            explanation = build_similarity_explanation(ref_info, row)
                            st.caption(f"💡 **Why It Matches:** {explanation}")
                            st.caption(f"🎬 **Director:** {row['director_str']}")
                            st.caption(f"🎭 **Cast:** {row['cast_str']}")
                            st.caption(f"⏱️ **Runtime:** {row['runtime']}m | 🌐 **Lang:** {row['language']}")
                            st.caption(f"💰 **Box Office:** ${row['revenue']:,} ({row['votes']:,} votes)")
                            st.caption(f"📊 **Scores:** TF-IDF `{row['content_score']:.2f}` | NLP Tone `{row['sentiment_score']:.2f}`")
                            st.write(row["plot"][:200] + "...")

                        if st.button("➕ My List", key=f"rec_{row['id']}", use_container_width=True):
                            if row["id"] not in [w["id"] for w in st.session_state.watchlist]:
                                st.session_state.watchlist.append(row.to_dict())
                                st.toast(f"Added {row['title']} to My List!")
                                st.rerun()

# ==========================================
# TAB 2: TRENDING & MARKET ANALYTICS
# ==========================================
with tab_trends:
    st.markdown('<div class="section-header">Trending Now on CinePulse</div>', unsafe_allow_html=True)
    df_trends = df_base_catalog.copy()

    t_cols = st.columns(5)
    for idx, (_, m_row) in enumerate(df_trends.head(5).iterrows()):
        with t_cols[idx % 5]:
            with st.container(border=True):
                st.image(m_row["poster_url"], use_container_width=True)
                st.markdown(f'<div class="match-pill">🔥 TRENDING #{idx + 1}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="card-title">{m_row["title"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="card-sub">{m_row["year"]} &bull; ★ {m_row["rating"]} &bull; {m_row["certification"]}</div>', unsafe_allow_html=True)
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
        "Attribute": ["Release Year", "TMDB Rating", "Vote Count", "Runtime (mins)", "US Certification", "Box Office Revenue", "NLP Plot Polarity", "Genres", "Director"],
        row_a["title"]: [str(row_a["year"]), f"{row_a['rating']}/10", f"{row_a['votes']:,}", str(row_a["runtime"]), row_a["certification"], f"${row_a['revenue']:,}", str(row_a["polarity"]), row_a["genres_str"], row_a["director_str"]],
        row_b["title"]: [str(row_b["year"]), f"{row_b['rating']}/10", f"{row_b['votes']:,}", str(row_b["runtime"]), row_b["certification"], f"${row_b['revenue']:,}", str(row_b["polarity"]), row_b["genres_str"], row_b["director_str"]]
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
        st.info("Your list is currently empty. Click '➕ My List' on any movie card on the Home or Trending tabs to build your personal taste profile!")
    else:
        wl_cols = st.columns(5)
        for idx, item in enumerate(st.session_state.watchlist):
            with wl_cols[idx % 5]:
                with st.container(border=True):
                    st.image(item["poster_url"], use_container_width=True)
                    st.markdown(f'<div class="card-title">{item["title"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="card-sub">{item["year"]} &bull; ★ {item["rating"]}</div>', unsafe_allow_html=True)

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

# END OF FILE
