"""
model.py — CineMatch v4 · Optimized Recommendation Engine
Algorithms: CF (time-decay) | KNN | SVD | Content | Hybrid | Serendipity
            GenreBlend | WatchParty | DirectorRecs | ProfileMatch | Trending (Wilson)
"""
import pandas as pd, numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
import warnings; warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────
def load_data(rp="data/ratings.csv", mp="data/movies.csv"):
    r = pd.read_csv(rp)
    m = pd.read_csv(mp)
    return r, m

def build_user_item_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    return ratings.pivot_table(index="userId", columns="movieId", values="rating")

def build_movie_lookup(movies_df: pd.DataFrame) -> dict:
    """Fast O(1) movie lookup by movieId."""
    return {int(row["movieId"]): row.to_dict() for _, row in movies_df.iterrows()}

# ─────────────────────────────────────────────────────────────────────────────
# 1. COLLABORATIVE FILTERING  (time-decay weighted)
# ─────────────────────────────────────────────────────────────────────────────
class CollaborativeFilter:
    def __init__(self, matrix: pd.DataFrame, ratings: pd.DataFrame):
        self.matrix = matrix
        # Time-decay: recent ratings weighted higher via timestamp column if available
        if "timestamp" in ratings.columns:
            t_max = ratings["timestamp"].max()
            ratings = ratings.copy()
            ratings["weight"] = np.exp(-0.0001 * (t_max - ratings["timestamp"]))
            weighted = ratings.copy()
            weighted["wr"] = weighted["rating"] * weighted["weight"]
            wm = weighted.pivot_table(index="userId", columns="movieId",
                                       values="wr", aggfunc="sum")
            wm_cnt = weighted.pivot_table(index="userId", columns="movieId",
                                           values="weight", aggfunc="sum")
            self._wmatrix = (wm / wm_cnt).reindex_like(matrix)
        else:
            self._wmatrix = matrix

        filled = self._wmatrix.fillna(0)
        sim    = cosine_similarity(filled)
        self.sim_df = pd.DataFrame(sim, index=matrix.index, columns=matrix.index)

    def similar_users(self, uid, n=20) -> list:
        if uid not in self.sim_df.index: return []
        return self.sim_df[uid].drop(uid).nlargest(n).index.tolist()

    def recommend(self, uid, movies_df, n=10, exclude_ids=None) -> pd.DataFrame:
        similar = self.similar_users(uid, 30)
        if not similar: return pd.DataFrame()
        seen = set(self.matrix.loc[uid].dropna().index)
        if exclude_ids: seen |= set(exclude_ids)
        scores = self._wmatrix.loc[similar].mean(axis=0)
        scores = scores.drop(index=list(seen & set(scores.index)), errors="ignore")
        top = scores.nlargest(n).index.tolist()
        out = movies_df[movies_df["movieId"].isin(top)].copy()
        out["score"]  = out["movieId"].map(scores)
        out["reason"] = "Loved by users with your taste"
        return out.sort_values("score", ascending=False).head(n)

    def similar_user_pct(self, uid, n=5) -> dict:
        if uid not in self.sim_df.index: return {}
        return (self.sim_df[uid].drop(uid).nlargest(n) * 100).round(1).to_dict()

# ─────────────────────────────────────────────────────────────────────────────
# 2. KNN ITEM-BASED
# ─────────────────────────────────────────────────────────────────────────────
class KNNRecommender:
    def __init__(self, matrix: pd.DataFrame):
        self.item_matrix = matrix.T.fillna(0)
        self.model = NearestNeighbors(
            metric="cosine", algorithm="brute",
            n_neighbors=min(25, len(self.item_matrix)))
        self.model.fit(csr_matrix(self.item_matrix.values))
        self._mid_index = {mid: i for i, mid in enumerate(self.item_matrix.index)}

    def recommend(self, movie_id, movies_df, n=10, exclude_ids=None) -> pd.DataFrame:
        if movie_id not in self._mid_index: return pd.DataFrame()
        idx  = self._mid_index[movie_id]
        vec  = self.item_matrix.iloc[idx].values.reshape(1, -1)
        k    = min(n + 1, len(self.item_matrix))
        dist, inds = self.model.kneighbors(vec, n_neighbors=k)
        sim_ids  = [self.item_matrix.index[i] for i in inds.flatten()[1:]]
        dist_map = dict(zip(sim_ids, dist.flatten()[1:]))
        if exclude_ids: sim_ids = [s for s in sim_ids if s not in set(exclude_ids)]
        out = movies_df[movies_df["movieId"].isin(sim_ids)].copy()
        out["similarity"] = out["movieId"].map(lambda m: round(1 - dist_map.get(m, 1), 4))
        out["reason"]     = "Attracts the same audience"
        return out.sort_values("similarity", ascending=False).head(n)

# ─────────────────────────────────────────────────────────────────────────────
# 3. SVD MATRIX FACTORIZATION
# ─────────────────────────────────────────────────────────────────────────────
class SVDRecommender:
    def __init__(self, matrix: pd.DataFrame, k: int = 60):
        self.matrix = matrix
        global_mean = matrix.stack().mean()
        filled      = matrix.fillna(global_mean)
        safe_k      = min(k, min(matrix.shape) - 1)
        U, s, Vt    = svds(filled.values.astype(np.float32), k=safe_k)
        self.pred_df = pd.DataFrame(
            np.dot(np.dot(U, np.diag(s)), Vt),
            index=matrix.index, columns=matrix.columns)

    def recommend(self, uid, movies_df, n=10, exclude_ids=None) -> pd.DataFrame:
        if uid not in self.pred_df.index: return pd.DataFrame()
        seen = set(self.matrix.loc[uid].dropna().index)
        if exclude_ids: seen |= set(exclude_ids)
        scores = self.pred_df.loc[uid].drop(
            index=list(seen & set(self.pred_df.columns)), errors="ignore")
        top = scores.nlargest(n).index.tolist()
        out = movies_df[movies_df["movieId"].isin(top)].copy()
        out["predicted_rating"] = out["movieId"].map(scores).round(2)
        out["reason"]           = "Matrix factorization prediction"
        return out.sort_values("predicted_rating", ascending=False).head(n)

# ─────────────────────────────────────────────────────────────────────────────
# 4. CONTENT-BASED (genre + language + decade features)
# ─────────────────────────────────────────────────────────────────────────────
class ContentBasedFilter:
    def __init__(self, movies_df: pd.DataFrame):
        self.movies_df = movies_df
        genre_d = movies_df["genres"].str.get_dummies(sep="|")
        lang_d  = movies_df["language"].str.get_dummies() if "language" in movies_df.columns else pd.DataFrame()
        decade  = ((movies_df["year"].fillna(2000).astype(int) // 10) * 10).astype(str)
        dec_d   = pd.get_dummies(decade, prefix="dec")
        # Combine features with weights
        feat = pd.concat([genre_d * 3, lang_d * 0.5, dec_d * 0.3], axis=1).fillna(0)
        feat.index = movies_df["movieId"]
        self.sim_df = pd.DataFrame(
            cosine_similarity(feat), index=movies_df["movieId"], columns=movies_df["movieId"])

    def recommend(self, movie_id, n=10, exclude_ids=None) -> pd.DataFrame:
        if movie_id not in self.sim_df.index: return pd.DataFrame()
        scores = self.sim_df[movie_id].drop(movie_id)
        if exclude_ids:
            scores = scores.drop(index=[e for e in exclude_ids if e in scores.index], errors="ignore")
        top = scores.nlargest(n).index.tolist()
        out = self.movies_df[self.movies_df["movieId"].isin(top)].copy()
        out["genre_similarity"] = out["movieId"].map(scores).round(4)
        out["reason"]           = "Shares genre, language & era DNA"
        return out.sort_values("genre_similarity", ascending=False).head(n)

# ─────────────────────────────────────────────────────────────────────────────
# 5. HYBRID ENSEMBLE (configurable weights)
# ─────────────────────────────────────────────────────────────────────────────
class HybridRecommender:
    def __init__(self, cf, knn, svd, cbf):
        self.cf, self.knn, self.svd, self.cbf = cf, knn, svd, cbf

    @staticmethod
    def _merge(frames: list, score_cols: list, weights: list,
               exclude_mid=None, keep_cols=None) -> pd.DataFrame:
        if not frames: return pd.DataFrame()
        base = keep_cols or ["movieId","title","genres","year","director",
                              "imdb_rating","runtime","cast","tagline","language"]
        agg_parts = []
        for frame, sc, w in zip(frames, score_cols, weights):
            if frame.empty: continue
            tmp = frame.copy()
            tmp["_s"] = tmp[sc].fillna(0) * w
            agg_parts.append(tmp[[c for c in base + ["_s"] if c in tmp.columns]])
        if not agg_parts: return pd.DataFrame()
        combined = pd.concat(agg_parts)
        available = [c for c in base if c in combined.columns and c != "movieId"]
        result = (combined.groupby("movieId")
                          .agg(**{c: (c, "first") for c in available},
                               hybrid_score=("_s", "sum"))
                          .reset_index())
        if exclude_mid:
            result = result[result["movieId"] != exclude_mid]
        result["reason"] = "Multi-algorithm consensus"
        return result

    def by_movie(self, mid, movies_df, n=10, exclude_ids=None,
                 w_knn=0.5, w_cbf=0.5) -> pd.DataFrame:
        r1 = self.knn.recommend(mid, movies_df, n=n*3, exclude_ids=exclude_ids)
        r2 = self.cbf.recommend(mid, n=n*3, exclude_ids=exclude_ids)
        merged = self._merge([r1, r2], ["similarity","genre_similarity"],
                              [w_knn, w_cbf], exclude_mid=mid)
        return merged.nlargest(n, "hybrid_score") if not merged.empty else pd.DataFrame()

    def for_user(self, uid, movies_df, n=10, exclude_ids=None,
                 w_cf=0.45, w_svd=0.55) -> pd.DataFrame:
        r1 = self.cf.recommend(uid, movies_df, n=n*3, exclude_ids=exclude_ids)
        r2 = self.svd.recommend(uid, movies_df, n=n*3, exclude_ids=exclude_ids)
        seen = set(self.cf.matrix.loc[uid].dropna().index
                   if uid in self.cf.matrix.index else [])
        if exclude_ids: seen |= set(exclude_ids)
        merged = self._merge([r1, r2], ["score","predicted_rating"], [w_cf, w_svd])
        if merged.empty: return pd.DataFrame()
        merged = merged[~merged["movieId"].isin(seen)]
        merged["reason"] = "Community + SVD consensus"
        return merged.nlargest(n, "hybrid_score")

# ─────────────────────────────────────────────────────────────────────────────
# 6. GENRE BLEND  (find perfect cross-genre films)
# ─────────────────────────────────────────────────────────────────────────────
def recommend_genre_blend(genres: list, movies_df: pd.DataFrame,
                          ratings_df: pd.DataFrame, n=12,
                          imdb_floor=6.5) -> pd.DataFrame:
    """Score films by how well they span ALL requested genres."""
    if not genres: return pd.DataFrame()
    avg = ratings_df.groupby("movieId")["rating"].agg(avg="mean", cnt="count").reset_index()
    scored = movies_df.merge(avg, on="movieId", how="left")
    scored["avg"] = scored["avg"].fillna(3.5)
    scored["cnt"] = scored["cnt"].fillna(0)

    def blend_score(row):
        mg    = set(row["genres"].split("|"))
        hits  = sum(1 for g in genres if g in mg)
        cover = hits / len(genres)  # 0→1: how many target genres present
        qual  = (row["avg"] - 3.0) / 2.0
        imdb  = (row["imdb_rating"] - 7.0) / 3.0 if pd.notna(row.get("imdb_rating")) else 0
        return cover * 2.0 + qual * 0.6 + imdb * 0.4

    scored["blend_score"] = scored.apply(blend_score, axis=1)
    scored = scored[scored["imdb_rating"].fillna(0) >= imdb_floor]
    scored["reason"] = "Perfect blend of " + " × ".join(genres)
    return scored.nlargest(n, "blend_score")

# ─────────────────────────────────────────────────────────────────────────────
# 7. WATCH PARTY (consensus for multiple users)
# ─────────────────────────────────────────────────────────────────────────────
def recommend_watch_party(user_ids: list, cf, svd, movies_df: pd.DataFrame,
                          n=10) -> pd.DataFrame:
    """Aggregate predictions across N users — find what everyone will enjoy."""
    if not user_ids: return pd.DataFrame()
    score_maps = []
    for uid in user_ids:
        r_cf  = cf.recommend(uid, movies_df, n=50)
        r_svd = svd.recommend(uid, movies_df, n=50)
        sm = {}
        if not r_cf.empty:
            for _, row in r_cf.iterrows():
                sm[int(row["movieId"])] = sm.get(int(row["movieId"]), 0) + row["score"]
        if not r_svd.empty:
            for _, row in r_svd.iterrows():
                mid = int(row["movieId"])
                sm[mid] = sm.get(mid, 0) + row["predicted_rating"]
        score_maps.append(sm)

    # Intersection-first: movies all users like, then union fallback
    all_movies = set(score_maps[0].keys())
    for sm in score_maps[1:]: all_movies &= sm.keys()
    if len(all_movies) < n:
        for sm in score_maps: all_movies |= sm.keys()

    party_scores = {}
    for mid in all_movies:
        total = sum(sm.get(mid, 0) for sm in score_maps)
        party_scores[mid] = total / len(score_maps)

    top = sorted(party_scores, key=party_scores.get, reverse=True)[:n]
    out = movies_df[movies_df["movieId"].isin(top)].copy()
    out["party_score"] = out["movieId"].map(party_scores)
    out["reason"] = f"Best pick for {len(user_ids)} different taste profiles"
    return out.sort_values("party_score", ascending=False)

# ─────────────────────────────────────────────────────────────────────────────
# 8. SERENDIPITY  (novelty-quality trade-off)
# ─────────────────────────────────────────────────────────────────────────────
def recommend_serendipity(uid, ratings_df, movies_df, diversity=0.7, n=10):
    if uid in ratings_df["userId"].values:
        seen_ids = set(ratings_df[ratings_df["userId"]==uid]["movieId"])
        user_genres = set()
        for mid in seen_ids:
            row = movies_df[movies_df["movieId"]==mid]
            if not row.empty:
                user_genres |= set(row.iloc[0]["genres"].split("|"))
    else:
        seen_ids, user_genres = set(), set()

    avg = (ratings_df.groupby("movieId")["rating"]
           .agg(avg="mean", cnt="count").reset_index().query("cnt>=5"))
    scored = movies_df.merge(avg, on="movieId", how="inner")
    scored = scored[~scored["movieId"].isin(seen_ids)]

    def ss(row):
        genres  = set(row["genres"].split("|"))
        novelty = 1 - len(genres & user_genres) / max(len(genres), 1)
        quality = (row["avg"] - 3.0) / 2.0
        return novelty * diversity + quality * (1 - diversity)

    scored["seren_score"] = scored.apply(ss, axis=1)
    scored["reason"]      = "🎲 Curated outside your comfort zone"
    return scored.nlargest(n, "seren_score")

# ─────────────────────────────────────────────────────────────────────────────
# 9. DIRECTOR-BASED
# ─────────────────────────────────────────────────────────────────────────────
def recommend_by_director(director: str, current_mid, movies_df, ratings_df, n=8):
    same = movies_df[
        (movies_df["director"].str.contains(director, case=False, na=False)) &
        (movies_df["movieId"] != current_mid)
    ].copy()
    avg = ratings_df.groupby("movieId")["rating"].mean().reset_index()
    same = same.merge(avg, on="movieId", how="left")
    same["rating"] = same["rating"].fillna(3.5)
    same["reason"] = f"From the same director: {director}"
    return same.sort_values("imdb_rating", ascending=False).head(n)

# ─────────────────────────────────────────────────────────────────────────────
# 10. PROFILE MATCH  (cold-start from explicit ratings)
# ─────────────────────────────────────────────────────────────────────────────
def recommend_from_profile(user_ratings: dict, movies_df, ratings_df, n=12):
    liked    = {m: r for m, r in user_ratings.items() if r >= 4.0}
    disliked = {m: r for m, r in user_ratings.items() if r <= 2.5}
    liked_g, dis_g = {}, {}
    for mid, r in liked.items():
        row = movies_df[movies_df["movieId"]==mid]
        if row.empty: continue
        w = r - 3.5
        for g in row.iloc[0]["genres"].split("|"):
            liked_g[g] = liked_g.get(g, 0) + w
    for mid, r in disliked.items():
        row = movies_df[movies_df["movieId"]==mid]
        if row.empty: continue
        for g in row.iloc[0]["genres"].split("|"):
            dis_g[g] = dis_g.get(g, 0) + (3.5 - r)
    exclude = set(user_ratings.keys())
    avg_r = ratings_df.groupby("movieId")["rating"].mean()

    def ps(row):
        gs  = row["genres"].split("|")
        pos = sum(liked_g.get(g, 0) for g in gs)
        neg = sum(dis_g.get(g, 0) for g in gs) * 0.6
        com = (avg_r.get(row["movieId"], 3.5) - 3.0) * 0.4
        return pos - neg + com

    out = movies_df[~movies_df["movieId"].isin(exclude)].copy()
    out["profile_score"] = out.apply(ps, axis=1)
    out["reason"]        = "Matched to your Taste DNA"
    return out.nlargest(n, "profile_score")

# ─────────────────────────────────────────────────────────────────────────────
# 11. MOOD PICKS
# ─────────────────────────────────────────────────────────────────────────────
MOOD_MAP = {
    "😤 Adrenaline Rush":   ["adrenaline","intense","action"],
    "🧠 Mind-Bending":      ["mind-bending","psychological"],
    "😢 Emotional Journey": ["emotional","feel-good","inspiring"],
    "😂 Something Fun":     ["fun","quirky","comedy"],
    "🌑 Dark & Gritty":     ["dark","gritty","noir"],
    "✨ Epic & Grand":      ["epic","inspiring"],
    "🌫️ Atmospheric":      ["atmospheric","slow-burn"],
    "🕵️ Mystery & Crime":  ["mystery","crime"],
    "❤️ Romance":           ["emotional","feel-good","romance"],
    "🤯 WTF Cinema":        ["mind-bending","dark","atmospheric"],
}

def recommend_by_mood(mood, movies_df, ratings_df, n=10):
    kws    = MOOD_MAP.get(mood, [])
    avg    = ratings_df.groupby("movieId")["rating"].agg(avg="mean", cnt="count").reset_index()
    scored = movies_df.merge(avg, on="movieId", how="left")
    scored["avg"] = scored["avg"].fillna(3.5)
    def ms(row):
        tags = str(row.get("mood_tags","")).lower()
        hits = sum(1 for k in kws if k in tags)
        return hits * 2.0 + (row["avg"] - 3.0) * 0.5
    scored["mood_score"] = scored.apply(ms, axis=1)
    scored["reason"]     = f"Hand-picked for '{mood}'"
    return scored[scored["mood_score"] > 0].nlargest(n, "mood_score")

# ─────────────────────────────────────────────────────────────────────────────
# 12. TRENDING  (Wilson score lower bound)
# ─────────────────────────────────────────────────────────────────────────────
def get_trending(ratings_df, movies_df, n=10, recent_pct=0.3):
    cutoff = int(len(ratings_df) * (1 - recent_pct))
    recent = ratings_df.iloc[cutoff:]
    stats  = recent.groupby("movieId")["rating"].agg(avg="mean", cnt="count").reset_index()
    # Wilson score-like: balance between quality and popularity
    z = 1.96  # 95% CI
    stats["wilson"] = (
        (stats["avg"] / 5 + z**2 / (2 * stats["cnt"]) -
         z * np.sqrt(stats["avg"] / 5 * (1 - stats["avg"] / 5) / stats["cnt"] + z**2 / (4 * stats["cnt"]**2)))
        / (1 + z**2 / stats["cnt"])
    )
    out = movies_df.merge(stats, on="movieId", how="inner")
    out["trend_score"] = out["wilson"]
    out["reason"]      = "Community trending pick 🔥"
    return out.nlargest(n, "trend_score")

# ─────────────────────────────────────────────────────────────────────────────
# 13. TIME MACHINE  (decade-based discovery)
# ─────────────────────────────────────────────────────────────────────────────
def recommend_time_machine(decade: int, movies_df, ratings_df, n=12):
    decade_movies = movies_df[
        (movies_df["year"].fillna(0).astype(int) // 10 * 10) == decade
    ].copy()
    avg = ratings_df.groupby("movieId")["rating"].agg(avg="mean", cnt="count").reset_index()
    out = decade_movies.merge(avg, on="movieId", how="left")
    out["avg"] = out["avg"].fillna(3.5)
    out["cnt"] = out["cnt"].fillna(0)
    out["time_score"] = out["avg"] * np.log1p(out["cnt"]) + out["imdb_rating"].fillna(7) * 0.3
    out["reason"]     = f"A gem from the {decade}s"
    return out.nlargest(n, "time_score")

# ─────────────────────────────────────────────────────────────────────────────
# 14. SMART SEARCH  (fuzzy multi-field)
# ─────────────────────────────────────────────────────────────────────────────
def smart_search(query: str, movies_df: pd.DataFrame, max_results=20) -> pd.DataFrame:
    q = query.lower().strip()
    if not q: return pd.DataFrame()

    def score(row):
        s = 0
        title    = str(row.get("title","")).lower()
        director = str(row.get("director","")).lower()
        cast_s   = str(row.get("cast","")).lower()
        tagline  = str(row.get("tagline","")).lower()
        genres   = str(row.get("genres","")).lower()
        mood     = str(row.get("mood_tags","")).lower()

        if q == title:                s += 100
        elif title.startswith(q):     s += 60
        elif q in title:              s += 40
        if q in director:             s += 30
        if q in cast_s:               s += 25
        if q in genres.replace("|"," "): s += 20
        if q in tagline:              s += 15
        if q in mood:                 s += 10
        # partial word match
        for word in q.split():
            if word in title:         s += 8
            if word in director:      s += 5
            if word in cast_s:        s += 4
        return s

    results = movies_df.copy()
    results["search_score"] = results.apply(score, axis=1)
    results = results[results["search_score"] > 0]
    results["reason"] = f"Matches '{query}'"
    return results.nlargest(max_results, "search_score")

# ─────────────────────────────────────────────────────────────────────────────
# 15. TASTE PROFILE
# ─────────────────────────────────────────────────────────────────────────────
def get_taste_profile(uid, ratings_df, movies_df) -> dict:
    user_r = ratings_df[ratings_df["userId"]==uid].merge(movies_df, on="movieId")
    if user_r.empty: return {}
    gs = {}
    for _, row in user_r.iterrows():
        for g in str(row["genres"]).split("|"):
            if g not in gs: gs[g] = {"total": 0, "count": 0}
            gs[g]["total"] += row["rating"]; gs[g]["count"] += 1
    return {g: round(v["total"]/v["count"], 2) for g, v in gs.items() if v["count"] >= 1}

# ─────────────────────────────────────────────────────────────────────────────
# 16. EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def precision_at_k(rec, rel, k=10):
    return len(set(rec[:k]) & set(rel)) / k if k else 0

def recall_at_k(rec, rel, k=10):
    return len(set(rec[:k]) & set(rel)) / len(rel) if rel else 0

def ndcg_at_k(rec, rel, k=10):
    rel_set = set(rel)
    dcg  = sum(1/np.log2(i+2) for i,m in enumerate(rec[:k]) if m in rel_set)
    idcg = sum(1/np.log2(i+2) for i in range(min(k, len(rel))))
    return dcg/idcg if idcg else 0

def run_evaluation(ratings, movies, cf, svd, sample_n=50):
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(ratings, test_size=0.2, random_state=42)
    users = test["userId"].unique()[:sample_n]
    out = {"Method":[],"P@10":[],"R@10":[],"NDCG@10":[]}
    for uid in users:
        rel = test[(test["userId"]==uid)&(test["rating"]>=4)]["movieId"].tolist()
        if not rel: continue
        for lbl, fn in [("CF", lambda u=uid: cf.recommend(u, movies, 10)),
                         ("SVD",lambda u=uid: svd.recommend(u, movies, 10))]:
            r = fn()
            if r.empty: continue
            ids = r["movieId"].tolist()
            out["Method"].append(lbl)
            out["P@10"].append(precision_at_k(ids, rel))
            out["R@10"].append(recall_at_k(ids, rel))
            out["NDCG@10"].append(ndcg_at_k(ids, rel))
    if not out["Method"]: return pd.DataFrame()
    return pd.DataFrame(out).groupby("Method").mean().round(4)
