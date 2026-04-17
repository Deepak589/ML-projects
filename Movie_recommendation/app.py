import streamlit as st
import pickle
import numpy as np
import pandas as pd

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Hybrid Movie Recommender",
    layout="wide"
)

st.title("🎬 Hybrid Movie Recommendation System")
st.caption("Content-Based Similarity + Collaborative Filtering (SVD)")

# -----------------------------
# CACHED LOADERS
# -----------------------------
@st.cache_resource
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_numpy(path):
    return np.load(path)

# -----------------------------
# LOAD ARTIFACTS
# -----------------------------
smd = load_pickle("artifacts/smd.pkl")                  # DataFrame
indices = load_pickle("artifacts/indices.pkl")          # title -> index
indices_map = load_pickle("artifacts/indices_map.pkl")  # tmdbId -> movieId
svd = load_pickle("artifacts/svd.pkl")                  # Surprise SVD
cosine_sim = load_pickle("artifacts/cosine_sim.pkl")

# -----------------------------
# HYBRID RECOMMENDER
# -----------------------------
def hybrid(user_id, title, topn=10, k=25):

    if title not in indices.index:
        return pd.DataFrame()

    idx_val = indices.loc[title]
    idx = int(idx_val.iloc[0] if hasattr(idx_val, "iloc") else idx_val)

    sim_scores = sorted(
        list(enumerate(cosine_sim[idx])),
        key=lambda x: x[1],
        reverse=True
    )[1 : k + 1]

    movie_indices = [i for i, _ in sim_scores]

    candidates = smd.iloc[movie_indices][
        ["title", "vote_count", "vote_average", "year", "id"]
    ].copy()

    preds = []
    for tmdb_id in candidates["id"]:
        try:
            movie_id = int(indices_map.loc[int(tmdb_id)])
            est = svd.predict(int(user_id), movie_id).est
        except Exception:
            est = np.nan
        preds.append(est)

    candidates["predicted_rating"] = preds
    candidates = candidates.dropna(subset=["predicted_rating"])

    return candidates.sort_values(
        "predicted_rating", ascending=False
    ).head(topn)


# -----------------------------
# STREAMLIT UI
# -----------------------------
col1, col2 = st.columns([1, 3])

with col1:
    user_id = st.number_input(
        "👤 User ID",
        min_value=1,
        value=1,
        step=1
    )

with col2:
    movie = st.selectbox(
        "🎥 Select a Movie",
        sorted(smd["title"].dropna().unique())
    )

if st.button("🚀 Recommend", use_container_width=True):
    with st.spinner("Finding the best matches for you..."):
        results = hybrid(user_id, movie)

    if results.empty:
        st.warning("No recommendations found for this movie/user.")
    else:
        st.success("Recommendations ready 🎉")
        st.dataframe(
            results.reset_index(drop=True),
            use_container_width=True
        )
