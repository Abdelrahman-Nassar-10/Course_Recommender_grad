import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches
from datetime import datetime

# ----------------- Page Config -----------------
st.set_page_config(page_title="Course Recommendation System", page_icon="🎓", layout="wide")
st.title("🎓 Course Recommendation System")

# ----------------- Sidebar: Data Path -----------------
st.sidebar.header("📦 Data")
default_csv_path = "combined_courses.csv"   # <- change if you keep it elsewhere
csv_path = st.sidebar.text_input("CSV file path", default_csv_path)

# ----------------- Load Data -----------------
@st.cache_resource(show_spinner=True)
def load_df(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)

    # normalize expected columns
    # required: course_title, url, level, Rating, subject
    for col in ["course_title", "url", "level", "Rating", "subject",
                "price", "num_subscribers", "num_reviews", "num_lectures",
                "content_duration", "published_timestamp", "course_id"]:
        if col not in df.columns:
            df[col] = np.nan

    # clean types
    for c in ["price", "num_subscribers", "num_reviews", "num_lectures", "content_duration", "Rating"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # parse year for filtering
    def to_year(x):
        try:
            return pd.to_datetime(x).year
        except Exception:
            return np.nan
    df["year"] = df["published_timestamp"].apply(to_year)

    # fill text nulls
    for c in ["course_title", "subject", "level"]:
        df[c] = df[c].fillna("").astype(str)

    # keep a clean index
    return df.reset_index(drop=True)

try:
    data = load_df(csv_path)
    st.sidebar.success(f"Loaded {len(data)} courses.")
except Exception as e:
    st.sidebar.error(str(e))
    st.stop()

# ----------------- Build Text & Vectorizer -----------------
@st.cache_resource(show_spinner=True)
def build_vectorizer(df: pd.DataFrame):
    # Combine fields to represent content (you can tweak weights if you like)
    text = (
        df["course_title"].str.strip() + " | " +
        df["subject"].str.strip() + " | " +
        df["level"].str.strip()
    ).fillna("")

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9
    )
    X = vectorizer.fit_transform(text.values)
    # L2-normalized by default; cosine_similarity on TF-IDF is effective here
    return vectorizer, X

vectorizer, tfidf_matrix = build_vectorizer(data)

# ----------------- Helpers -----------------
def candidate_filter_idx(
    df: pd.DataFrame,
    subjects_sel,
    levels_sel,
    min_rating,
    price_max,
    min_reviews,

) -> np.ndarray:
    mask = np.ones(len(df), dtype=bool)

    if subjects_sel:
        mask &= df["subject"].isin(subjects_sel)
    if levels_sel:
        mask &= df["level"].isin(levels_sel)
    if min_rating is not None:
        mask &= (df["Rating"].fillna(0) >= float(min_rating))
    if price_max is not None:
        mask &= (df["price"].fillna(0) <= float(price_max))
    if min_reviews is not None:
        mask &= (df["num_reviews"].fillna(0) >= float(min_reviews))
    # if year_min is not None:
    #     mask &= (df["year"].fillna(-1) >= int(year_min))
    # if year_max is not None:
    #     mask &= (df["year"].fillna(10**6) <= int(year_max))

    return np.where(mask)[0]

def find_course_indices_by_name(df: pd.DataFrame, typed: str) -> list[int]:
    names = df["course_title"].fillna("").tolist()
    matches = get_close_matches(typed, names, n=10, cutoff=0.3)
    if matches:
        idxs = []
        for m in matches:
            idxs.extend(df.index[df["course_title"] == m].tolist())
        return idxs
    # fallback: substring
    contains = df["course_title"].fillna("").str.lower().str.contains(typed.lower())
    return df[contains].index.tolist()

def mmr_diversify(sorted_idx: np.ndarray, sims: np.ndarray, k: int = 5, lambda_: float = 0.75) -> list:
    selected = []
    candidates = list(sorted_idx)
    sim_lookup = {idx: s for idx, s in zip(sorted_idx, sims)}

    while candidates and len(selected) < k:
        if not selected:
            best = candidates.pop(0)
            selected.append(best)
            continue
        best_idx, best_score = None, -1e9
        for c in candidates[:250]:
            rel = sim_lookup[c]
            div = 0.0
            for s in selected:
                # cosine between course vectors; tfidf_matrix is L2-normalized internally
                div = max(div, float(tfidf_matrix[c].dot(tfidf_matrix[s].T).toarray()[0][0]))
            score = lambda_ * rel - (1 - lambda_) * div
            if score > best_score:
                best_score, best_idx = score, c
        candidates.remove(best_idx)
        selected.append(best_idx)
    return selected

def top_k_from_query_idx(
    query_idx: int,
    candidate_idx: np.ndarray,
    k: int,
    strategy: str = "similarity",
    use_mmr: bool = True
) -> list[tuple[int, float]]:
    qvec = tfidf_matrix[query_idx]
    sims = cosine_similarity(qvec, tfidf_matrix[candidate_idx]).ravel()
    order = np.argsort(sims)[::-1]
    ranked = candidate_idx[order]
    sims_sorted = sims[order]

    # drop the course itself if present
    keep = ranked != query_idx
    ranked = ranked[keep]
    sims_sorted = sims_sorted[keep]

    if strategy == "quality":
        # simple quality bump: include Rating and log(1+num_reviews)
        ratings = data.loc[ranked, "Rating"].fillna(0).to_numpy(float)
        reviews = np.log1p(data.loc[ranked, "num_reviews"].fillna(0).to_numpy(float))
        score = 0.75 * sims_sorted + 0.15 * (ratings / 5.0) + 0.10 * (reviews / (reviews.max() + 1e-8))
        rerank = np.argsort(score)[::-1]
        ranked = ranked[rerank]
        sims_sorted = sims_sorted[rerank]

    if use_mmr and len(ranked) > k:
        chosen = mmr_diversify(ranked, sims_sorted, k=k, lambda_=0.75)
        sim_map = {i: s for i, s in zip(ranked, sims_sorted)}
        return [(i, float(sim_map[i])) for i in chosen[:k]]
    else:
        return [(int(i), float(s)) for i, s in zip(ranked[:k], sims_sorted[:k])]

# ----------------- Sidebar: Filters & Options -----------------
st.sidebar.header("🔧 Filters & Options")

subjects = sorted([s for s in data["subject"].dropna().unique().tolist() if str(s).strip()])
levels = sorted([l for l in data["level"].dropna().unique().tolist() if str(l).strip()])

f_subjects = st.sidebar.multiselect("Subject(s)", subjects, default=[])
f_levels = st.sidebar.multiselect("Level(s)", levels, default=[])

min_rating = st.sidebar.slider("Min Rating", 0.0, 1.0)
price_max = st.sidebar.number_input("Max Price", min_value=0.0, value=200.0, step=10.0)
min_reviews = st.sidebar.number_input("Min #Reviews", min_value=0, value=50, step=10)

# yr_min_default = int(np.nanmin(data["year"])) if pd.notna(data["year"]).any() else 2010
# yr_max_default = int(np.nanmax(data["year"])) if pd.notna(data["year"]).any() else 2025
# year_min = st.sidebar.number_input("Year From", min_value=2000, max_value=2100, value=yr_min_default, step=1)
# year_max = st.sidebar.number_input("Year To", min_value=2000, max_value=2100, value=yr_max_default, step=1)

top_n = st.sidebar.slider("Number of recommendations", 3, 20, 7)
rank_strategy = st.sidebar.selectbox("Ranking strategy", ["similarity", "quality"])
use_mmr = st.sidebar.toggle("Diversify similar results (MMR)", value=True)

# ----------------- Tabs -----------------
tab_search, tab_browse = st.tabs(["🔎 Search & Recommend", "📚 Browse Catalog"])

with tab_search:
    st.subheader("Search for a course")
    typed_course = st.text_input("Start typing a course title:")

    suggested_display = None
    suggested_indices = []
    if typed_course.strip():
        idxs = find_course_indices_by_name(data, typed_course.strip())
        if idxs:
            options = []
            for i in idxs:
                row = data.iloc[i]
                name = row["course_title"] or "Untitled"
                subj = row["subject"] or "—"
                lvl = row["level"] or "—"
                options.append(f"{name} — {subj} — {lvl}  [#{i}]")
            suggested_display = st.selectbox("Did you mean:", options)
            suggested_indices = idxs
        else:
            st.info("No close matches found. Try a different keyword or browse.")

    candidate_idx = candidate_filter_idx(
        data, f_subjects, f_levels, min_rating, price_max, min_reviews
    )
    # st.caption(f"🎯 Candidates that match filters: **{len(candidate_idx)}** / {len(data)}")

    if st.button("Get Recommendations"):
        if not suggested_display:
            st.error("Please type a title and pick one suggestion.")
        elif len(candidate_idx) == 0:
            st.warning("No courses match current filters.")
        else:
            try:
                q_idx = int(suggested_display.split("[#")[-1].rstrip("]"))
            except Exception:
                st.error("Could not parse selected course index. Try again.")
                st.stop()

            results = top_k_from_query_idx(
                q_idx, candidate_idx, k=top_n, strategy=rank_strategy, use_mmr=use_mmr
            )

            qrow = data.iloc[q_idx]
            st.success(f"Top {len(results)} recommendations for **{qrow['course_title']}**:")

            for rank, (rid, sim) in enumerate(results, start=1):
                row = data.iloc[rid]
                with st.container(border=True):
                    st.markdown(f"**{rank}. {row.get('course_title', 'Untitled')}**")
                    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
                    with c1: st.caption(f"Subject: {row.get('subject', '—')}")
                    with c2: st.caption(f"Level: {row.get('level', '—')}")
                    with c3: st.caption(f"Rating: {row.get('Rating', '—')}")
                    with c4: st.caption(f"Similarity: {sim:.3f}")
                    # show light metadata
                    c5, c6, c7 = st.columns([1,1,1])
                    with c5: st.caption(f"Price: {row.get('price', '—')}")
                    with c6: st.caption(f"Reviews: {row.get('num_reviews', '—')}")
                    with c7: st.caption(f"Subscribers: {row.get('num_subscribers', '—')}")
                    url = row.get("url", "")
                    if isinstance(url, str) and url.strip():
                        st.markdown(f"[Open Course]({url})")

with tab_browse:
    st.subheader("Browse Catalog")
    col1, col2 = st.columns([1, 1])
    sort_options = ["course_title", "subject", "level", "Rating", "num_reviews", "num_subscribers", "price", "content_duration", "year"]
    with col1:
        sort_key = st.selectbox("Sort by", sort_options, index=sort_options.index("Rating"))
    with col2:
        asc = st.toggle("Ascending", value=False)

    browse_idx = candidate_filter_idx(
        data, f_subjects, f_levels, min_rating, price_max, min_reviews
    )
    view = data.iloc[browse_idx].copy()
    if sort_key in view.columns:
        view = view.sort_values(sort_key, ascending=asc)

    st.write(f"Showing {len(view)} courses")
    st.dataframe(
        view[["course_title", "subject", "level", "Rating", "price", "num_reviews", "num_subscribers", "url", "year"]]
        .reset_index(drop=True),
        use_container_width=True,
        hide_index=True
    )
