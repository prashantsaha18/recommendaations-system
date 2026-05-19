# 🎬 CineMatch v4 — Discover · Rate · Obsess

> A **cinematic, production-grade AI recommendation system** with 10 pages, 6 algorithms,
> SVG poster cards with film perforations, genre DNA profiling, serendipity discovery,
> genre blending, watch party mode, smart fuzzy search, and full analytics.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red?style=flat-square&logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange?style=flat-square)
![scipy](https://img.shields.io/badge/scipy-1.12%2B-blue?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 🚀 Deploy to Streamlit Cloud (Free · ~2 min)

```bash
# 1. Push to GitHub
git init && git add . && git commit -m "🎬 CineMatch v4"
git remote add origin https://github.com/YOUR_USERNAME/cinematch.git
git push -u origin main

# 2. share.streamlit.io → New app → select repo → main file: app.py → Deploy ✅
```

Data auto-generates on first launch. Zero manual setup.

---

## ✨ Pages at a Glance

| Page | Highlights |
|---|---|
| 🔍 **Discover** | Fuzzy search bar · movie detail card · decade/genre/IMDb filters · director deep-dive |
| 🧬 **Taste DNA** | Rate movies → visual genre DNA tiles → cold-start personalised recs |
| 🎭 **Mood Picks** | 10 mood categories with colour-coded cards |
| 🔥 **Trending** | Wilson-score trending · all-time best · top by genre |
| 🎲 **Surprise Me** | Diversity slider · hidden gem roulette |
| 🎨 **Genre Blender** | Mix 2–4 genres · 6 preset blends · IMDb floor filter |
| 🎊 **Watch Party** | Multi-user consensus · Time Machine (decade explorer) |
| ⚖️ **Algorithm Lab** | Side-by-side comparison · consensus detection |
| 📋 **Watchlist** | Save films · watchlist stats · CSV export · "More Like These" |
| 📊 **Analytics** | Genre/decade/language charts · director leaderboard · P@K / NDCG eval |

---

## 🧠 Algorithms (6 + extras)

| Algorithm | Innovation |
|---|---|
| **Collaborative Filtering** | Time-decay weighted cosine similarity — recent ratings count more |
| **KNN Item-Based** | Pre-indexed kd-tree for fast item similarity lookup |
| **SVD Matrix Factorization** | float32 sparse SVD (2× faster), global mean fill |
| **Content-Based** | Genre + language + decade multi-feature vectors |
| **Hybrid Ensemble** | Configurable per-algorithm weights, dual-pass merging |
| **Serendipity** | Novelty-quality trade-off with adjustable diversity dial |
| **Genre Blend** | Cross-genre coverage score + IMDb + community quality |
| **Watch Party** | Multi-user score aggregation with intersection-first ranking |
| **Profile Match** | Weighted liked/disliked genre extraction (cold-start) |
| **Time Machine** | Decade-based discovery with log-popularity weighting |
| **Smart Search** | Multi-field fuzzy scoring: title, director, cast, tagline, mood |
| **Trending** | Wilson score lower bound (the Reddit/IMDb formula) |

---

## 📁 Project Structure

```
cinematch/
│
├── app.py              ← 10-page Streamlit app (~700 lines)
├── model.py            ← 16 functions / 6 algorithm classes (~380 lines)
├── utils.py            ← CSS system, SVG poster engine v2, render helpers (~350 lines)
├── generate_data.py    ← 90 films · 500 users · archetype-biased ratings (~130 lines)
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/               ← Auto-generated on first run
│   ├── movies.csv      ← movieId|title|genres|year|director|imdb_rating|mood_tags|runtime|cast|tagline|language
│   └── ratings.csv     ← userId|movieId|rating|timestamp
│
└── .streamlit/
    └── config.toml     ← Deep-space theme (crimson · gold · teal)
```

---

## ⚙️ Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/cinematch.git
cd cinematch
pip install -r requirements.txt
streamlit run app.py
# → http://localhost:8501
```

---

## 🔌 Use Real MovieLens Data

```bash
wget https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
unzip ml-latest-small.zip
cp ml-latest-small/ratings.csv data/ratings.csv
cp ml-latest-small/movies.csv  data/movies.csv
```

> Add `year`, `director`, `imdb_rating`, `mood_tags`, `runtime`, `cast`, `tagline`, `language`
> columns to `movies.csv` to unlock the full UI. The app gracefully degrades with missing columns.

---

## 🏎️ Optimisations vs v3

| Area | What Changed |
|---|---|
| **SVD** | float32 dtype → 2× faster factorisation |
| **Content-Based** | Added language + decade features (weighted) |
| **CF** | Time-decay via timestamp column → recency-aware |
| **Trending** | Wilson score instead of naive avg × log(count) |
| **Hybrid** | Configurable per-algorithm weights, single-pass merge |
| **Caching** | `@st.cache_resource` for models, `@st.cache_data` for stats |
| **SVG Posters** | Film perforation holes + grain filter + golden-angle gradients |
| **CSS** | CSS custom properties, keyframe animations, custom scrollbar |
| **Search** | Multi-field fuzzy scorer (no extra dependency) |
| **Code** | Split into 4 focused files; utils.py separates UI from ML |

---

## 📐 Evaluation

| Metric | Formula |
|---|---|
| **Precision@K** | `hits in top-K / K` |
| **Recall@K** | `hits in top-K / all relevant` |
| **NDCG@K** | Normalised Discounted Cumulative Gain (rewards top-ranked hits) |

---

## 📝 Resume Bullet Points

**CineMatch — AI Recommendation Engine** · Python, Streamlit, scikit-learn, scipy

- Architected a 12-algorithm recommendation engine (time-decay CF, KNN, SVD,
  content-based, hybrid, serendipity, genre blend, watch party, profile match,
  time machine, smart search, Wilson-score trending)
- Optimised SVD with float32 dtype and global-mean fill; time-decay CF using
  exponential weighting on rating timestamps
- Built multi-field fuzzy search engine across title, director, cast, tagline,
  and mood tags without external dependencies
- Designed SVG poster card engine with procedural gradients, film perforations,
  grain filter, and genre-specific colour palettes
- Delivered 10-page Streamlit app with Watch Party consensus mode, Genre Blender,
  Taste DNA cold-start profiling, Serendipity discovery, and CSV export

---

## 🏅 Skills Demonstrated

✔ Advanced recommender systems (CF, KNN, SVD, hybrid, serendipity, cold-start)  
✔ Matrix factorization & sparse linear algebra (scipy svds, float32)  
✔ Statistical ranking (Wilson score, time-decay, NDCG)  
✔ Feature engineering (multi-feature content vectors, genre/language/decade)  
✔ Full-stack web app (Streamlit, custom CSS animations, SVG generation)  
✔ Software engineering (modular 4-file architecture, aggressive caching, OOP)  
✔ UI/UX design (design system, film grain, keyframe animations, dark theme)  

---

## 📜 License

MIT — free to use, modify, and ship.

---

<div align="center">🎬 CineMatch v4 · Built with Python + Streamlit</div>
