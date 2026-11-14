import streamlit as st
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import plotly.express as px

# ==========================================
# 1. PAGE CONFIGURATION & BRANDING
# ==========================================
st.set_page_config(
    page_title="CinePulse AI | Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# API Key setup using Streamlit Secrets
if "TMDB_KEY" in st.secrets:
    API_KEY = st.secrets["TMDB_KEY"]
else:
    st.error("API Key not found in Streamlit Secrets.")
    st.stop()

# Initialize session state for the User Watchlist (Enhanced Feature)
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

    # Extract Genres, Keywords, Cast, and Director
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
    poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None

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
        "poster_url": poster_url
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

    # Fetch Popular and Top Rated pages to ensure diverse filtering candidates
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

    return pd.DataFrame(records)

@st.cache_data(show_spinner=False, ttl=3600)
def search_tmdb_movie(query):
    """Searches TMDB by movie title."""
    url = "https://api.themoviedb.org/3/search/movie"
    res = requests.get(url, params={"api_key": API_KEY, "query": query}).json()
    return res.get("results", [])

# ==========================================
# 3. CORE FILTERING & RECOMMENDATION LOGIC
# ==========================================
def apply_core_filters(df, year_range, min_rating, min_votes, runtime_range, certs, lang, personnel_query, selected_genres, genre_op, exclude_id=None):
    """Applies all 5 required core filtering capabilities from the assignment rubric."""
    filtered = df.copy()
    if exclude_id is not None:
        filtered = filtered[filtered["id"] != exclude_id]

    # 1. Temporal Filters (Year range)
    filtered = filtered[(filtered["year"] >= year_range[0]) & (filtered["year"] <= year_range[1])]

    # 2. Quality Thresholds (Minimum rating and vote count)
    filtered = filtered[(filtered["rating"] >= min_rating) & (filtered["votes"] >= min_votes)]

    # 3. Content Specifications (Runtime, Certification, Language)
    filtered = filtered[(filtered["runtime"] >= runtime_range[0]) & (filtered["runtime"] <= runtime_range[1])]
    if certs:
        filtered = filtered[filtered["certification"].isin(certs)]
    if lang != "ALL":
        filtered = filtered[filtered["language"] == lang]

    # 4. Personnel Filter (Actor or Director preference)
    if personnel_query.strip():
        q = personnel_query.strip().lower()
        filtered = filtered[
            filtered["cast_str"].str.lower().str.contains(q, na=False) |
            filtered["director_str"].str.lower().str.contains(q, na=False)
        ]

    # 5. Genre Combinations with Logical Operators (AND / OR)
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
        reasons.append(f"Shares genres: {', '.join(shared_genres)}")

    ref_cast = set(reference_info.get("cast", []))
    cand_cast = set(candidate_row.get("cast", []))
    shared_cast = ref_cast.intersection(cand_cast)
    if shared_cast:
        reasons.append(f"Shared cast: {', '.join(shared_cast)}")

    ref_kw = set(reference_info.get("keywords", []))
    cand_kw = set(candidate_row.get("keywords", []))
    shared_kw = ref_kw.intersection(cand_kw)
    if shared_kw:
        reasons.append(f"Matching themes/keywords: {', '.join(list(shared_kw)[:3])}")

    pol_diff = abs(reference_info.get("polarity", 0) - candidate_row.get("polarity", 0))
    if pol_diff < 0.15:
        reasons.append("Highly similar narrative sentiment and emotional tone")

    if not reasons:
        reasons.append("Strong TF-IDF plot vocabulary alignment and high audience quality score")

    return " | ".join(reasons)

# ==========================================
# 4. SIDEBAR: ALL 5 CORE FILTERING CAPABILITIES
# ==========================================
st.sidebar.title("🎛️ Core Filters & Settings")
st.sidebar.caption("Use these controls to test all required filtering capabilities in combination.")

rec_approach = st.sidebar.selectbox(
    "Recommendation Algorithm",
    [
        "Hybrid (Content + NLP Sentiment + Quality)",
        "Content-Based Filtering (TF-IDF Attributes)",
        "Sentiment-Based NLP Matching (Plot Tone)"
    ]
)

st.sidebar.subheader("1. Temporal Filters")
year_range = st.sidebar.slider("Release Year Range", 1970, 2026, (1995, 2026))

st.sidebar.subheader("2. Quality Thresholds")
min_rating = st.sidebar.slider("Minimum TMDB Rating", 0.0, 10.0, 6.0, 0.5)
min_votes = st.sidebar.number_input("Minimum Vote Count", min_value=0, max_value=10000, value=50, step=50)

st.sidebar.subheader("3. Content Specifications")
runtime_range = st.sidebar.slider("Runtime Bracket (mins)", 40, 240, (70, 210))
all_certs = ["G", "PG", "PG-13", "R", "NR"]
selected_certs = st.sidebar.multiselect("Certification Levels (US)", all_certs, default=all_certs)
selected_lang = st.sidebar.selectbox("Original Language", ["ALL", "EN", "FR", "ES", "JA", "KO"])

st.sidebar.subheader("4. Personnel Preferences")
personnel_query = st.sidebar.text_input("Preferred Actor or Director", placeholder="e.g., Keanu Reeves")

st.sidebar.subheader("5. Genre Logical Combinations")
genre_options = ["Action", "Adventure", "Animation", "Comedy", "Crime", "Drama", "Family", "Fantasy", "Horror", "Mystery", "Romance", "Science Fiction", "Thriller"]
selected_genres = st.sidebar.multiselect("Select Genres", genre_options)
genre_operator = st.sidebar.radio("Genre Logical Operator", ["Match ANY selected genre (OR)", "Match ALL selected genres (AND)"])

# ==========================================
# 5. MAIN BRANDED HEADER & NAVIGATION TABS
# ==========================================
st.title("🎬 CinePulse AI: Intelligent Movie Discovery Engine")
st.markdown("**ENTR 3901 Movie Recommendation System** | Powered by Live TMDB Metadata, TF-IDF Feature Extraction, and NLP Sentiment Analysis")

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Recommendations & NLP Query",
    "🔥 Trending & Data Visualizations",
    "⚖️ Comparative Movie Analysis",
    f"⭐ My Watchlist ({len(st.session_state.watchlist)})"
])

# ==========================================
# TAB 1: RECOMMENDATIONS & NATURAL LANGUAGE QUERY
# ==========================================
with tab1:
    st.subheader("Discover Personalized Movie Recommendations")
    search_mode = st.radio(
        "Select Discovery Mode:",
        ["Search by Favorite Movie Title", "Natural Language Query Interface (Describe a Vibe/Plot)"],
        horizontal=True
    )

    if search_mode == "Search by Favorite Movie Title":
        user_input = st.text_input("Enter a movie title you enjoy:", value="The Matrix")
    else:
        user_input = st.text_input(
            "Describe the plot, themes, or mood you want to watch:",
            value="A futuristic cyberpunk hacker discovers reality is a computer simulation and fights back"
        )

    if st.button("Generate Recommendations", type="primary"):
        if not user_input.strip():
            st.warning("Please enter a movie title or natural language description before searching.")
        else:
            with st.spinner("Fetching live TMDB metadata, running NLP vectorization, and applying filters..."):
                seed_movie = None
                if search_mode == "Search by Favorite Movie Title":
                    search_results = search_tmdb_movie(user_input)
                    if not search_results:
                        st.error("No matching movie found on TMDB. Please try a different title.")
                        st.stop()
                    seed_id = search_results[0]["id"]
                    seed_movie = fetch_movie_details(seed_id)
                    df_pool = build_candidate_pool(seed_movie_id=seed_id)
                else:
                    df_pool = build_candidate_pool(seed_movie_id=None)

                # Compute TF-IDF Matrix across candidate pool
                vectorizer = TfidfVectorizer(stop_words="english")
                tfidf_matrix = vectorizer.fit_transform(df_pool["combined_features"])

                if seed_movie is not None:
                    st.info(
                        f"**Reference Movie:** {seed_movie['title']} ({seed_movie['year']}) | "
                        f"**Genres:** {seed_movie['genres_str']} | **Director:** {seed_movie['director_str']} | "
                        f"**Plot Sentiment Polarity:** {seed_movie['polarity']}"
                    )
                    target_vec = vectorizer.transform([seed_movie["combined_features"]])
                    ref_polarity = seed_movie["polarity"]
                    ref_subjectivity = seed_movie["subjectivity"]
                    ref_info = seed_movie
                    exclude_id = seed_movie["id"]
                else:
                    q_blob = TextBlob(user_input)
                    ref_polarity = round(q_blob.sentiment.polarity, 3)
                    ref_subjectivity = round(q_blob.sentiment.subjectivity, 3)
                    st.info(
                        f"**Natural Language Query Analyzed** | "
                        f"**Detected Sentiment Polarity:** {ref_polarity} | **Subjectivity:** {ref_subjectivity}"
                    )
                    target_vec = vectorizer.transform([user_input])
                    ref_info = {"genres": selected_genres, "cast": [], "keywords": user_input.lower().split(), "polarity": ref_polarity}
                    exclude_id = None

                # 1. Content-Based Similarity Score (Cosine Similarity)
                content_scores = cosine_similarity(target_vec, tfidf_matrix).flatten()
                df_pool["content_score"] = content_scores

                # 2. NLP Sentiment Similarity Score
                pol_diff = (df_pool["polarity"] - ref_polarity).abs()
                sub_diff = (df_pool["subjectivity"] - ref_subjectivity).abs()
                df_pool["sentiment_score"] = (1.0 - ((pol_diff + sub_diff) / 3.0)).clip(0, 1)

                # 3. Select Algorithm Score
                if rec_approach.startswith("Content-Based"):
                    df_pool["final_score"] = df_pool["content_score"]
                elif rec_approach.startswith("Sentiment-Based"):
                    df_pool["final_score"] = (0.75 * df_pool["sentiment_score"]) + (0.25 * df_pool["content_score"])
                else:
                    norm_rating = df_pool["rating"] / 10.0
                    df_pool["final_score"] = (0.55 * df_pool["content_score"]) + (0.25 * df_pool["sentiment_score"]) + (0.20 * norm_rating)

                # Apply all 5 Sidebar Filters
                filtered_df = apply_core_filters(
                    df_pool, year_range, min_rating, min_votes, runtime_range,
                    selected_certs, selected_lang, personnel_query, selected_genres,
                    genre_operator, exclude_id=exclude_id
                )

                filtered_df = filtered_df.sort_values(by="final_score", ascending=False)

                if filtered_df.empty:
                    st.warning("No movies matched all of your active sidebar filters. Try widening the release year range, lowering minimum rating, or clearing personnel/genre filters.")
                else:
                    st.success(f"Found {len(filtered_df)} matching movies! Showing top recommendations below:")
                    for _, row in filtered_df.head(6).iterrows():
                        with st.container(border=True):
                            col1, col2 = st.columns([1, 4])
                            with col1:
                                if row["poster_url"]:
                                    st.image(row["poster_url"], use_container_width=True)
                                else:
                                    st.write("No Poster Available")
                            with col2:
                                st.markdown(f"### {row['title']} ({row['year']}) — Match Score: `{row['final_score']:.2f}`")
                                st.caption(
                                    f"⭐ **Rating:** {row['rating']}/10 ({row['votes']:,} votes) | "
                                    f"⏱️ **Runtime:** {row['runtime']} mins | "
                                    f"🔞 **Cert:** {row['certification']} | "
                                    f"🌐 **Lang:** {row['language']} | "
                                    f"💰 **Box Office:** ${row['revenue']:,}"
                                )
                                st.write(f"**Genres:** {row['genres_str']} | **Director:** {row['director_str']} | **Top Cast:** {row['cast_str']}")
                                st.write(row["plot"])
                                explanation = build_similarity_explanation(ref_info, row)
                                st.markdown(f"💡 **Similarity Explanation:** *{explanation}*")
                                st.caption(f"📊 Breakdown: Content TF-IDF = `{row['content_score']:.2f}` | NLP Sentiment Alignment = `{row['sentiment_score']:.2f}` (Plot Polarity: `{row['polarity']}`)")

# ==========================================
# TAB 2: TRENDING & DATA VISUALIZATIONS
# ==========================================
with tab2:
    st.subheader("Live TMDB Trending Movies & Market Analytics")
    df_trends = build_candidate_pool()
    col_a, col_b = st.columns(2)

    with col_a:
        fig_scatter = px.scatter(
            df_trends,
            x="rating",
            y="popularity",
            size="votes",
            color="decade",
            hover_name="title",
            hover_data=["year", "genres_str", "revenue"],
            title="Audience Rating vs. Popularity Score (Sized by Vote Count)"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_b:
        fig_sent = px.scatter(
            df_trends,
            x="polarity",
            y="subjectivity",
            color="rating",
            hover_name="title",
            title="NLP Plot Sentiment Map: Polarity vs. Subjectivity"
        )
        st.plotly_chart(fig_sent, use_container_width=True)

    st.markdown("#### Quick-Add Trending Movies to Your Watchlist")
    cols = st.columns(4)
    for idx, (_, m_row) in enumerate(df_trends.head(8).iterrows()):
        with cols[idx % 4]:
            if m_row["poster_url"]:
                st.image(m_row["poster_url"], use_container_width=True)
            st.markdown(f"**{m_row['title']}** ({m_row['year']})")
            st.caption(f"⭐ {m_row['rating']} | {m_row['genres_str']}")
            if st.button(f"➕ Add to Watchlist", key=f"trend_{m_row['id']}"):
                if m_row["id"] not in [w["id"] for w in st.session_state.watchlist]:
                    st.session_state.watchlist.append(m_row.to_dict())
                    st.toast(f"Added {m_row['title']} to your Watchlist!")
                    st.rerun()

# ==========================================
# TAB 3: COMPARATIVE MOVIE ANALYSIS
# ==========================================
with tab3:
    st.subheader("Side-by-Side Comparative Movie Analysis")
    df_comp = build_candidate_pool()
    movie_titles = df_comp["title"].tolist()

    c1, c2 = st.columns(2)
    with c1:
        movie_a_title = st.selectbox("Select First Movie", movie_titles, index=0)
    with c2:
        default_b = 1 if len(movie_titles) > 1 else 0
        movie_b_title = st.selectbox("Select Second Movie", movie_titles, index=default_b)

    row_a = df_comp[df_comp["title"] == movie_a_title].iloc[0]
    row_b = df_comp[df_comp["title"] == movie_b_title].iloc[0]

    comp_table = pd.DataFrame({
        "Metric": ["Release Year", "TMDB Rating", "Vote Count", "Runtime (mins)", "US Certification", "Box Office Revenue", "NLP Plot Polarity", "Genres", "Director"],
        row_a["title"]: [str(row_a["year"]), f"{row_a['rating']}/10", f"{row_a['votes']:,}", str(row_a["runtime"]), row_a["certification"], f"${row_a['revenue']:,}", str(row_a["polarity"]), row_a["genres_str"], row_a["director_str"]],
        row_b["title"]: [str(row_b["year"]), f"{row_b['rating']}/10", f"{row_b['votes']:,}", str(row_b["runtime"]), row_b["certification"], f"${row_b['revenue']:,}", str(row_b["polarity"]), row_b["genres_str"], row_b["director_str"]]
    })
    st.table(comp_table)

# ==========================================
# TAB 4: USER WATCHLIST & PROFILE RECOMMENDATIONS
# ==========================================
with tab4:
    st.subheader("Your Saved Watchlist & Profile-Based Recommendations")
    add_title = st.text_input("Quickly add any movie by title to your Watchlist:", placeholder="e.g., Inception")
    if st.button("Search & Add to Watchlist") and add_title.strip():
        res = search_tmdb_movie(add_title.strip())
        if res:
            details = fetch_movie_details(res[0]["id"])
            if details and details["id"] not in [w["id"] for w in st.session_state.watchlist]:
                st.session_state.watchlist.append(details)
                st.success(f"Added {details['title']} to your Watchlist!")
                st.rerun()

    if not st.session_state.watchlist:
        st.info("Your watchlist is currently empty. Add movies above or from the Trending tab to generate profile-based recommendations!")
    else:
        st.write("### Saved Watchlist Movies:")
        for item in st.session_state.watchlist:
            st.markdown(f"- **{item['title']} ({item['year']})** | Genres: {item['genres_str']} | Rating: ⭐ {item['rating']}")

        if st.button("Clear Watchlist"):
            st.session_state.watchlist = []
            st.rerun()

        st.divider()
        st.write("### 🎯 Profile-Based Recommendations (Generated from Your Watchlist)")
        df_all = build_candidate_pool()
        watchlist_ids = [w["id"] for w in st.session_state.watchlist]
        combined_profile_text = " ".join([w["combined_features"] for w in st.session_state.watchlist])

        vec = TfidfVectorizer(stop_words="english")
        mat = vec.fit_transform(df_all["combined_features"])
        prof_vec = vec.transform([combined_profile_text])

        df_all["profile_score"] = cosine_similarity(prof_vec, mat).flatten()
        recs = df_all[~df_all["id"].isin(watchlist_ids)].sort_values(by="profile_score", ascending=False).head(4)

        for _, r in recs.iterrows():
            st.markdown(f"**{r['title']} ({r['year']})** — Profile Match Score: `{r['profile_score']:.2f}` | ⭐ {r['rating']} | *{r['genres_str']}*")
