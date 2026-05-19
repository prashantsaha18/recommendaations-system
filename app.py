"""
🎬 CineMatch v4 — Discover · Rate · Obsess
10 pages · 6 algorithms · SVG posters · Genre Blender · Watch Party · Smart Search
"""
import streamlit as st
import pandas as pd
import numpy as np
import os, sys, subprocess, io

st.set_page_config(page_title="CineMatch", page_icon="🎬",
                   layout="wide", initial_sidebar_state="expanded")

from utils import inject_css, hero, sh, genre_tags_html, score_bar_html
from utils import render_poster_grid, render_list, GENRE_PALETTE, MOOD_PALETTE
inject_css()

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
DEFAULTS = {
    "watchlist": [], "not_interested": [], "my_ratings": {},
    "active_mood": None, "compare_results": {},
    "profile_ratings": {}, "search_results": None,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
# DATA & MODELS  (cached aggressively)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_all():
    if not os.path.exists("data/ratings.csv"):
        subprocess.run([sys.executable, "generate_data.py"], check=True)
    r = pd.read_csv("data/ratings.csv")
    m = pd.read_csv("data/movies.csv")
    return r, m

@st.cache_resource(show_spinner=False)
def build_models(_r, _m):
    from model import (build_user_item_matrix, build_movie_lookup,
                       CollaborativeFilter, KNNRecommender,
                       SVDRecommender, ContentBasedFilter, HybridRecommender)
    mx  = build_user_item_matrix(_r)
    cf  = CollaborativeFilter(mx, _r)
    knn = KNNRecommender(mx)
    svd = SVDRecommender(mx, k=60)
    cbf = ContentBasedFilter(_m)
    hyb = HybridRecommender(cf, knn, svd, cbf)
    lkp = build_movie_lookup(_m)
    return mx, cf, knn, svd, cbf, hyb, lkp

@st.cache_data(show_spinner=False)
def get_avg_ratings(_r):
    return _r.groupby("movieId")["rating"].agg(avg="mean", cnt="count").reset_index()

# ─────────────────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("🎬 Warming up the projector…"):
    ratings, movies = load_all()
    mx, cf, knn, svd, cbf, hyb, movie_lookup = build_models(ratings, movies)
    avg_ratings = get_avg_ratings(ratings)

from model import (recommend_by_mood, recommend_serendipity, get_trending,
                   recommend_by_director, recommend_from_profile,
                   get_taste_profile, recommend_genre_blend,
                   recommend_watch_party, recommend_time_machine,
                   smart_search, run_evaluation, MOOD_MAP)

excl   = st.session_state.not_interested
titles = sorted(movies["title"].tolist())
all_genres = sorted({g for gs in movies["genres"].str.split("|") for g in gs})
all_directors = sorted(movies["director"].dropna().unique().tolist())
decades = sorted({int(y)//10*10 for y in movies["year"].dropna().astype(int)}, reverse=True)
all_users = sorted(ratings["userId"].unique().tolist())

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="text-align:center;padding:8px 0 2px">'
        '<span style="font-family:\'Bebas Neue\',sans-serif;font-size:1.5rem;'
        'letter-spacing:5px;background:linear-gradient(135deg,#f72585,#ffd166);'
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent">CINEMATCH</span>'
        '</div>', unsafe_allow_html=True)

    page = st.radio("", [
        "🔍 Discover", "🧬 Taste DNA", "🎭 Mood Picks",
        "🔥 Trending", "🎲 Surprise Me", "🎨 Genre Blender",
        "🎊 Watch Party", "⚖️ Algorithm Lab",
        "📋 Watchlist", "📊 Analytics",
    ], label_visibility="collapsed")

    st.markdown("---")
    n_recs    = st.slider("Results", 4, 16, 8)
    view_mode = st.radio("View", ["🖼️ Posters", "📄 List"], horizontal=True)
    use_grid  = view_mode == "🖼️ Posters"
    renderer  = render_poster_grid if use_grid else render_list

    st.markdown("---")
    n_m = len(movies); n_u = ratings["userId"].nunique()
    n_r = len(ratings); n_wl = len(st.session_state.watchlist)
    st.markdown(f"""<div class="stat-grid">
      <div class="stat-box"><div class="stat-num" style="color:#f72585">{n_m}</div>
        <div class="stat-lbl">Movies</div></div>
      <div class="stat-box"><div class="stat-num" style="color:#ffd166">{n_u}</div>
        <div class="stat-lbl">Users</div></div>
      <div class="stat-box"><div class="stat-num" style="color:#06d6a0">{n_r:,}</div>
        <div class="stat-lbl">Ratings</div></div>
      <div class="stat-box"><div class="stat-num" style="color:#9b8ef0">{n_wl}</div>
        <div class="stat-lbl">Saved</div></div>
    </div>""", unsafe_allow_html=True)

    if excl:
        st.markdown(f'<div style="margin-top:8px;padding:7px 12px;background:rgba(255,255,255,.02);'
                    f'border-radius:9px;font-size:.7rem;color:rgba(234,234,245,.35);">'
                    f'🚫 {len(excl)} excluded</div>', unsafe_allow_html=True)
        if st.button("Clear exclusions", use_container_width=True):
            st.session_state.not_interested = []; st.rerun()

# HERO
hero()

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: DISCOVER
# ═════════════════════════════════════════════════════════════════════════════
if page == "🔍 Discover":
    sh("🔍", "Discover")

    # Smart search bar at the top
    q = st.text_input("🔎 Search by title, director, cast, tagline…", placeholder="e.g. Nolan, Keanu, 'free your mind'")
    if q:
        results = smart_search(q, movies, max_results=n_recs * 2)
        results = results[~results["movieId"].isin(excl)]
        if not results.empty:
            sh("🔎", f'Results for "{q}"')
            renderer(results.head(n_recs), score_col="search_score", extra_key="srch")
        else:
            st.info(f'No results for "{q}" — try a different term.')
        st.markdown("---")

    tab_movie, tab_user = st.tabs(["🎬 By Movie", "👤 By User"])

    # ── BY MOVIE ──────────────────────────────────────────────────────────────
    with tab_movie:
        col_sel, col_flt = st.columns([3, 2])
        with col_sel:
            sel    = st.selectbox("Pick a movie you loved:", titles, key="disc_sel")
            picked = movies[movies["title"] == sel].iloc[0]
            mid    = int(picked["movieId"])
        with col_flt:
            dec_opts = ["All"] + [f"{d}s" for d in decades]
            dec_sel  = st.selectbox("Decade", dec_opts, key="disc_dec")
            gen_flt  = st.multiselect("Filter genres", all_genres, key="disc_gen")
            imdb_min = st.slider("Min IMDb", 5.0, 9.5, 6.0, 0.5, key="disc_imdb")

        # Movie detail card
        from utils import poster_svg
        avg_u  = ratings[ratings["movieId"] == mid]["rating"].mean()
        avg_t  = f"{avg_u:.1f} ★" if not np.isnan(avg_u) else "—"
        p_info = GENRE_PALETTE.get(picked["genres"].split("|")[0],
                                   {"fg":"#aaa","bg":"#111","ac":"#888"})
        svg_d  = poster_svg(picked["title"], picked.get("genres",""),
                            picked.get("year",""), picked.get("imdb_rating",""),
                            picked.get("runtime",""))

        c1, c2 = st.columns([1, 4])
        with c1:
            st.markdown(f'<div style="border-radius:12px;overflow:hidden;max-width:130px">{svg_d}</div>',
                        unsafe_allow_html=True)
        with c2:
            yr   = str(int(float(picked["year"]))) if pd.notna(picked.get("year")) else "—"
            rt   = f"{int(picked['runtime'])}min" if pd.notna(picked.get("runtime")) else ""
            cast = str(picked.get("cast","") or "")[:55]
            tl   = str(picked.get("tagline","") or "")
            lang = str(picked.get("language","") or "").upper()
            st.markdown(f"""
<div style="background:rgba(255,255,255,.025);border:1px solid {p_info['ac']}33;
  border-radius:14px;padding:16px 18px">
  <div style="font-family:'Bebas Neue',sans-serif;font-size:1.35rem;
    letter-spacing:2px;color:{p_info['ac']};margin-bottom:4px">{picked['title']}</div>
  <div style="font-size:.7rem;font-family:monospace;color:rgba(234,234,245,.4);margin-bottom:7px">
    {yr} · {rt} · ⭐ {picked.get('imdb_rating','—')} IMDb · 👥 {avg_t} community · {lang}
  </div>
  {genre_tags_html(picked.get('genres',''), 5)}
  {'<div style="font-size:.78rem;color:rgba(234,234,245,.45);font-style:italic;margin-top:8px">&#34;' + tl + '&#34;</div>' if tl else ''}
  {'<div style="font-size:.7rem;color:rgba(234,234,245,.3);margin-top:5px">🎬 ' + cast + '</div>' if cast else ''}
</div>""", unsafe_allow_html=True)

        algo = st.selectbox("Algorithm", [
            "🔀 Hybrid", "📐 KNN", "🎭 Content-Based"], key="disc_algo")

        b1, b2, b3 = st.columns(3)
        with b1: run     = st.button("🚀 Get Recommendations", type="primary", use_container_width=True)
        with b2: run_dir = st.button(f"🎥 {picked.get('director','')[:18]}'s Films", use_container_width=True)
        with b3:
            in_wl = mid in st.session_state.watchlist
            if st.button("✓ Saved" if in_wl else "+ Watchlist", use_container_width=True):
                if in_wl: st.session_state.watchlist.remove(mid)
                else:     st.session_state.watchlist.append(mid)
                st.rerun()

        if run:
            with st.spinner("Finding your next obsession…"):
                if "KNN"     in algo: res = knn.recommend(mid, movies, n=n_recs*2, exclude_ids=excl); sc="similarity"
                elif "Content" in algo: res = cbf.recommend(mid, n=n_recs*2, exclude_ids=excl);       sc="genre_similarity"
                else:                   res = hyb.by_movie(mid, movies, n=n_recs*2, exclude_ids=excl); sc="hybrid_score"
                if gen_flt:
                    res = res[res["genres"].apply(lambda g: any(f in g.split("|") for f in gen_flt))]
                if dec_sel != "All":
                    dec = int(dec_sel[:-1])
                    res = res[res["year"].apply(lambda y: pd.notna(y) and int(float(y))//10*10 == dec)]
                if imdb_min > 5.0:
                    res = res[res["imdb_rating"].fillna(0) >= imdb_min]
                sh("✨", f'Because you liked "{sel}"')
                renderer(res.head(n_recs), sc, extra_key="disc_m")

        if run_dir:
            dir_name = str(picked.get("director",""))
            if dir_name:
                with st.spinner(f"Pulling {dir_name}'s filmography…"):
                    dr = recommend_by_director(dir_name, mid, movies, ratings, n=10)
                    if not dr.empty:
                        sh("🎥", f"More from {dir_name}")
                        renderer(dr, "imdb_rating", max_score=10.0, extra_key="dir")
                    else:
                        st.info(f"No other films by {dir_name} in catalogue.")

    # ── BY USER ────────────────────────────────────────────────────────────────
    with tab_user:
        uid    = st.selectbox("User ID:", all_users, key="disc_uid")
        algo_u = st.selectbox("Algorithm", ["🔀 Hybrid", "🤝 Collaborative", "🔮 SVD"], key="disc_algou")
        taste  = get_taste_profile(uid, ratings, movies)
        hist   = (ratings[ratings["userId"] == uid]
                  .merge(movies, on="movieId")
                  .sort_values("rating", ascending=False))

        c_h, c_t = st.columns([1, 1])
        with c_h:
            with st.expander(f"📽️ User {uid}'s top-rated", expanded=True):
                for _, r in hist.head(10).iterrows():
                    stars = "★" * int(r["rating"]) + "☆" * (5 - int(r["rating"]))
                    st.markdown(
                        f'<div style="padding:5px 0;border-bottom:1px solid rgba(255,255,255,.04)">'
                        f'<span style="font-weight:600;font-size:.82rem">{r["title"]}</span><br>'
                        f'<span style="color:#ffd166;font-size:.72rem">{stars} {r["rating"]}</span>'
                        f'</div>', unsafe_allow_html=True)

        with c_t:
            if taste:
                st.markdown("**🧬 Genre affinity**")
                for g, sc in sorted(taste.items(), key=lambda x: -x[1])[:8]:
                    p   = GENRE_PALETTE.get(g, {"fg":"#aaa","ac":"#888"})
                    pct = int(sc / 5 * 100)
                    st.markdown(
                        f'<div style="display:flex;align-items:center;gap:9px;margin-bottom:5px">'
                        f'<span style="min-width:68px;font-size:.72rem;color:rgba(234,234,245,.6)">{g}</span>'
                        f'<div style="flex:1;height:5px;background:rgba(255,255,255,.06);border-radius:3px">'
                        f'<div style="width:{pct}%;height:5px;border-radius:3px;background:{p["ac"]}"></div></div>'
                        f'<span style="font-family:monospace;font-size:.66rem;color:{p["fg"]}">{sc:.1f}</span>'
                        f'</div>', unsafe_allow_html=True)

        if st.button("🚀 Personalise My Feed", type="primary", use_container_width=True):
            with st.spinner("Computing…"):
                if   "SVD"   in algo_u: res = svd.recommend(uid, movies, n=n_recs, exclude_ids=excl); sc="predicted_rating"
                elif "Collab" in algo_u: res = cf.recommend(uid, movies, n=n_recs, exclude_ids=excl); sc="score"
                else:                    res = hyb.for_user(uid, movies, n=n_recs, exclude_ids=excl); sc="hybrid_score"
                sh("🎯", "Picked For You")
                renderer(res, sc, extra_key="user_disc")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: TASTE DNA
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🧬 Taste DNA":
    sh("🧬", "Build Your Taste Profile")
    st.markdown('<p style="color:rgba(234,234,245,.42);font-size:.85rem;margin-bottom:1.2rem">'
                'Rate movies you\'ve seen. We\'ll decode your DNA and recommend films '
                'you\'ll love — no account needed.</p>', unsafe_allow_html=True)

    # Add movie to rate
    c_pick, c_add = st.columns([4, 1])
    with c_pick:
        rate_title = st.selectbox("Search a movie to rate:", titles, key="rate_sel")
    with c_add:
        st.markdown("<div style='margin-top:24px'/>", unsafe_allow_html=True)
        if st.button("Add ＋", use_container_width=True):
            row = movies[movies["title"] == rate_title].iloc[0]
            if int(row["movieId"]) not in st.session_state.profile_ratings:
                st.session_state.profile_ratings[int(row["movieId"])] = 3.5

    # Rating sliders
    to_rm = []
    if st.session_state.profile_ratings:
        st.markdown("#### ⭐ Your Ratings")
        for mid_r, cur in list(st.session_state.profile_ratings.items()):
            m = movies[movies["movieId"] == mid_r]
            if m.empty: continue
            m = m.iloc[0]
            ca, cb, cc = st.columns([3, 2, 1])
            with ca:
                st.markdown(f'<span style="font-weight:600;font-size:.88rem">{m["title"]}</span>'
                            f'<br>{genre_tags_html(m.get("genres",""), 2)}',
                            unsafe_allow_html=True)
            with cb:
                nv = st.select_slider("", [0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0],
                                      value=cur, key=f"rsl_{mid_r}",
                                      label_visibility="collapsed")
                st.session_state.profile_ratings[mid_r] = nv
            with cc:
                if st.button("✕", key=f"rmr_{mid_r}"): to_rm.append(mid_r)
    for m in to_rm:
        del st.session_state.profile_ratings[m]
        st.rerun()

    n_rated = len(st.session_state.profile_ratings)
    st.markdown(f'<p style="color:rgba(234,234,245,.35);font-size:.76rem">'
                f'{n_rated} rated — 3+ for best results.</p>', unsafe_allow_html=True)

    col_run, col_clr = st.columns([3, 1])
    with col_run:
        run_dna = st.button("🧬 Decode & Recommend", type="primary",
                            use_container_width=True, disabled=n_rated < 1)
    with col_clr:
        if st.button("🗑️ Reset", use_container_width=True):
            st.session_state.profile_ratings = {}; st.rerun()

    if run_dna and n_rated >= 1:
        with st.spinner("Decoding your taste DNA…"):
            result = recommend_from_profile(
                st.session_state.profile_ratings, movies, ratings, n=n_recs * 2)
            result = result[~result["movieId"].isin(excl)]

        # Genre DNA visualisation
        liked = {m: r for m, r in st.session_state.profile_ratings.items() if r >= 4.0}
        genre_aff: dict = {}
        for mid_r, r in liked.items():
            row = movies[movies["movieId"] == mid_r]
            if row.empty: continue
            for g in row.iloc[0]["genres"].split("|"):
                genre_aff[g] = genre_aff.get(g, 0) + (r - 3.0)

        if genre_aff:
            sh("🧬", "Your Taste DNA")
            total = sum(genre_aff.values()) or 1
            sorted_ga = sorted(genre_aff.items(), key=lambda x: -x[1])[:8]
            cols = st.columns(min(4, len(sorted_ga)))
            for i, (g, s) in enumerate(sorted_ga):
                pct = s / total * 100
                p   = GENRE_PALETTE.get(g, {"fg":"#aaa","bg":"#111","ac":"#888","icon":"🎬"})
                with cols[i % len(cols)]:
                    st.markdown(
                        f'<div style="background:{p["bg"]};border:1px solid {p["ac"]}44;'
                        f'border-radius:12px;padding:12px;text-align:center;margin-bottom:8px;'
                        f'animation:fadeSlideIn .4s ease {i*0.07:.2f}s both">'
                        f'<div style="font-size:1.7rem">{p["icon"]}</div>'
                        f'<div style="font-weight:700;font-size:.82rem;color:{p["ac"]};margin:4px 0">{g}</div>'
                        f'<div style="font-size:1rem;font-weight:800;font-family:monospace;'
                        f'color:white">{pct:.0f}%</div></div>', unsafe_allow_html=True)

        sh("🎯", "Recommendations Based on Your DNA")
        renderer(result.head(n_recs), "profile_score", extra_key="dna")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: MOOD PICKS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🎭 Mood Picks":
    sh("🎭", "What's Your Mood Tonight?")
    st.markdown('<p style="color:rgba(234,234,245,.42);font-size:.85rem;margin-bottom:1rem">'
                'Stop scrolling. Tell us how you feel.</p>', unsafe_allow_html=True)

    moods = list(MOOD_MAP.keys())
    cols  = st.columns(5)
    for i, mood in enumerate(moods):
        c = MOOD_PALETTE.get(mood, "#aaa")
        active = st.session_state.active_mood == mood
        with cols[i % 5]:
            border = f"1px solid {c}" if active else "1px solid rgba(255,255,255,.09)"
            bg     = f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16):.0f},.15)" if active else "rgba(255,255,255,.03)"
            st.markdown(
                f'<div style="background:{bg};border:{border};border-radius:11px;'
                f'padding:10px 6px;text-align:center;margin-bottom:8px;cursor:pointer;'
                f'font-size:.82rem;font-weight:600;color:{"white" if active else "rgba(234,234,245,.6)"}">'
                f'{mood}</div>', unsafe_allow_html=True)
            if st.button("Select", key=f"mood_{i}", use_container_width=True,
                         label_visibility="collapsed"):
                st.session_state.active_mood = mood; st.rerun()

    if st.session_state.active_mood:
        mood = st.session_state.active_mood
        c    = MOOD_PALETTE.get(mood, "#aaa")
        st.markdown(
            f'<div style="background:rgba(255,255,255,.025);border:1px solid {c}44;'
            f'border-radius:12px;padding:12px 16px;margin:8px 0 1rem">'
            f'<span style="font-size:1rem;font-weight:700;color:{c}">{mood}</span>'
            f'<span style="font-size:.78rem;color:rgba(234,234,245,.35);margin-left:10px">'
            f'Films hand-picked for this feeling</span></div>', unsafe_allow_html=True)
        result = recommend_by_mood(mood, movies, ratings, n=n_recs * 2)
        result = result[~result["movieId"].isin(excl)]
        renderer(result.head(n_recs), "mood_score", extra_key="mood")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: TRENDING
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔥 Trending":
    t1, t2, t3 = st.tabs(["🔥 Trending Now", "🏆 All-Time Best", "🌍 By Genre"])
    with t1:
        sh("🔥", "Trending Now")
        trend = get_trending(ratings, movies, n=n_recs * 2)
        trend = trend[~trend["movieId"].isin(excl)]
        renderer(trend.head(n_recs), "trend_score", extra_key="trend")
    with t2:
        sh("🏆", "All-Time Community Favourites")
        at = (avg_ratings.query("cnt >= 8").merge(movies, on="movieId")
              .nlargest(n_recs * 2, "avg"))
        at = at[~at["movieId"].isin(excl)]
        at["score"] = at["avg"]; at["reason"] = "All-time community favourite"
        renderer(at.head(n_recs), "score", max_score=5.0, extra_key="alltime")
    with t3:
        sh("🌍", "Best by Genre")
        sel_g = st.selectbox("Genre:", all_genres, key="trend_genre")
        top_g = (movies[movies["genres"].str.contains(sel_g, na=False)]
                 .merge(avg_ratings, on="movieId", how="left")
                 .sort_values("imdb_rating", ascending=False)
                 .head(n_recs * 2))
        top_g = top_g[~top_g["movieId"].isin(excl)]
        top_g["score"]  = top_g["imdb_rating"]
        top_g["reason"] = f"Top {sel_g} film"
        renderer(top_g.head(n_recs), "score", max_score=10.0, extra_key="genre_top")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: SURPRISE ME
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🎲 Surprise Me":
    sh("🎲", "Serendipity Mode")
    st.markdown('<p style="color:rgba(234,234,245,.42);font-size:.85rem;margin-bottom:1rem">'
                'Break your filter bubble. Discover something genuinely unexpected.</p>',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        uid_s = st.selectbox("Based on User (optional):",
                             ["Anonymous"] + all_users, key="seren_uid")
    with c2:
        diversity = st.slider("🔀 Diversity", 0.1, 1.0, 0.7, 0.05,
                              help="Higher = further outside your usual taste")

    if st.button("🎲 Surprise Me!", type="primary", use_container_width=True):
        uid_v = uid_s if uid_s != "Anonymous" else -1
        res   = recommend_serendipity(uid_v, ratings, movies, diversity=diversity, n=n_recs * 2)
        res   = res[~res["movieId"].isin(excl)]
        sh("🎲", "Your Serendipitous Picks")
        renderer(res.head(n_recs), "seren_score", extra_key="seren")

    st.markdown("---")
    sh("💎", "Hidden Gem Roulette")
    st.markdown('<p style="color:rgba(234,234,245,.38);font-size:.8rem">'
                'One random, underrated, critically acclaimed film.</p>',
                unsafe_allow_html=True)
    if st.button("🎰 Roll the Dice", use_container_width=True):
        gems = (movies.merge(avg_ratings.query("cnt>=5 and cnt<=30"), on="movieId")
                .query("imdb_rating >= 7.5"))
        gems = gems[~gems["movieId"].isin(excl)]
        if not gems.empty:
            gem = gems.sample(1).iloc[0]
            from utils import poster_svg
            svg_g = poster_svg(gem["title"], gem.get("genres",""),
                               gem.get("year",""), gem.get("imdb_rating",""),
                               gem.get("runtime",""))
            ca, cb = st.columns([1, 4])
            with ca:
                st.markdown(f'<div style="border-radius:11px;overflow:hidden;max-width:120px">'
                            f'{svg_g}</div>', unsafe_allow_html=True)
            with cb:
                yr = str(int(float(gem["year"]))) if pd.notna(gem.get("year")) else "—"
                st.markdown(f'**{gem["title"]}** ({yr})  \n'
                            f'{genre_tags_html(gem.get("genres",""),4)}  \n'
                            f'⭐ IMDb {gem.get("imdb_rating","—")} · 🎬 {gem.get("director","—")}  \n'
                            f'*"{gem.get("tagline","") or ""}"*',
                            unsafe_allow_html=True)
                gid = int(gem["movieId"])
                if gid not in st.session_state.watchlist:
                    if st.button("+ Add to Watchlist", key="gem_wl"):
                        st.session_state.watchlist.append(gid); st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: GENRE BLENDER
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🎨 Genre Blender":
    sh("🎨", "Genre Blender")
    st.markdown('<p style="color:rgba(234,234,245,.42);font-size:.85rem;margin-bottom:1rem">'
                'Mix genres and find films that span them all. Try Action + Comedy, '
                'or Sci-Fi + Romance + Mystery.</p>', unsafe_allow_html=True)

    sel_genres = st.multiselect("Pick 2–4 genres to blend:", all_genres,
                                default=["Action","Comedy"], max_selections=4)
    imdb_floor = st.slider("Min IMDb", 5.0, 9.0, 6.5, 0.5, key="blend_imdb")

    if sel_genres:
        # Show genre colour swatches
        swatch_html = ""
        for g in sel_genres:
            p = GENRE_PALETTE.get(g, {"fg":"#aaa","bg":"#222","ac":"#888","icon":"🎬"})
            swatch_html += (f'<span style="display:inline-flex;align-items:center;gap:6px;'
                            f'padding:5px 12px;border-radius:999px;margin-right:6px;'
                            f'background:{p["bg"]};border:1px solid {p["ac"]}55;'
                            f'color:{p["fg"]};font-weight:700;font-size:.8rem">'
                            f'{p["icon"]} {g}</span>')
        st.markdown(f'<div style="margin:8px 0 12px">{swatch_html}</div>',
                    unsafe_allow_html=True)

    if st.button("🎨 Blend & Discover", type="primary", use_container_width=True,
                 disabled=len(sel_genres) < 2):
        with st.spinner("Mixing genres…"):
            result = recommend_genre_blend(sel_genres, movies, ratings,
                                           n=n_recs * 2, imdb_floor=imdb_floor)
            result = result[~result["movieId"].isin(excl)]
            sh("✨", " × ".join(sel_genres))
            renderer(result.head(n_recs), "blend_score", extra_key="blend")

    # Preset blends
    st.markdown("---")
    st.markdown("#### ⚡ Quick Blends")
    presets = [
        ("🔥 Action + Comedy", ["Action","Comedy"]),
        ("🌌 Sci-Fi + Romance", ["Sci-Fi","Romance"]),
        ("🕵️ Crime + Thriller + Mystery", ["Crime","Thriller","Mystery"]),
        ("😢 Drama + Music", ["Drama","Music"]),
        ("👻 Horror + Comedy", ["Horror","Comedy"]),
        ("🤖 Sci-Fi + Action + Drama", ["Sci-Fi","Action","Drama"]),
    ]
    pc = st.columns(3)
    for i, (label, genres) in enumerate(presets):
        with pc[i % 3]:
            if st.button(label, use_container_width=True, key=f"preset_{i}"):
                res = recommend_genre_blend(genres, movies, ratings, n=n_recs)
                res = res[~res["movieId"].isin(excl)]
                sh("✨", " × ".join(genres))
                renderer(res, "blend_score", extra_key=f"preset_{i}")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: WATCH PARTY
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🎊 Watch Party":
    sh("🎊", "Watch Party Mode")
    st.markdown('<p style="color:rgba(234,234,245,.42);font-size:.85rem;margin-bottom:1rem">'
                'Picking a movie everyone will enjoy? Add multiple users and find '
                'the perfect consensus film.</p>', unsafe_allow_html=True)

    party_users = st.multiselect("Add users to the party:", all_users,
                                 default=all_users[:3], max_selections=8)

    # Show each user's taste
    if party_users:
        c_cols = st.columns(min(4, len(party_users)))
        for i, uid_p in enumerate(party_users):
            taste_p = get_taste_profile(uid_p, ratings, movies)
            top3    = sorted(taste_p.items(), key=lambda x: -x[1])[:3]
            badges  = " ".join(
                GENRE_PALETTE.get(g, {"icon":"🎬"})["icon"] + " " + g
                for g, _ in top3)
            with c_cols[i % len(c_cols)]:
                st.markdown(
                    f'<div style="background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.07);'
                    f'border-radius:11px;padding:12px;text-align:center;margin-bottom:8px">'
                    f'<div style="font-family:monospace;font-size:.72rem;color:rgba(234,234,245,.4)">User {uid_p}</div>'
                    f'<div style="font-size:.76rem;margin-top:4px;color:#eaeaf5">{badges}</div>'
                    f'</div>', unsafe_allow_html=True)

    if st.button("🎊 Find Party Picks", type="primary",
                 use_container_width=True, disabled=len(party_users) < 2):
        with st.spinner(f"Finding consensus for {len(party_users)} people…"):
            result = recommend_watch_party(party_users, cf, svd, movies, n=n_recs)
            result = result[~result["movieId"].isin(excl)]
            sh("🎊", f"Everyone Will Love These ({len(party_users)} users)")
            renderer(result, "party_score", extra_key="party")

    st.markdown("---")
    sh("⏰", "Time Machine Mode")
    st.markdown('<p style="color:rgba(234,234,245,.42);font-size:.82rem">'
                'Explore the best films from any decade.</p>', unsafe_allow_html=True)
    dec = st.select_slider("Pick a decade:", options=[d for d in decades],
                           value=decades[len(decades)//2],
                           format_func=lambda x: f"{x}s")
    if st.button("⏰ Take Me There", use_container_width=True):
        res = recommend_time_machine(dec, movies, ratings, n=n_recs)
        res = res[~res["movieId"].isin(excl)]
        sh("⏰", f"Best of the {dec}s")
        renderer(res, "time_score", extra_key="time")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: ALGORITHM LAB
# ═════════════════════════════════════════════════════════════════════════════
elif page == "⚖️ Algorithm Lab":
    sh("⚖️", "Algorithm Comparison Lab")
    st.markdown('<p style="color:rgba(234,234,245,.42);font-size:.85rem;margin-bottom:1rem">'
                'Run all algorithms side-by-side and see where they agree — or wildly disagree.</p>',
                unsafe_allow_html=True)

    mode = st.radio("Compare by:", ["🎬 Movie", "👤 User"], horizontal=True)

    if mode == "🎬 Movie":
        cmp_t = st.selectbox("Movie:", titles, key="cmp_t")
        cmp_m = int(movies[movies["title"] == cmp_t].iloc[0]["movieId"])
        top_n = st.slider("Top N", 3, 8, 5, key="cmp_n")
        if st.button("⚡ Run All Algorithms", type="primary", use_container_width=True):
            with st.spinner("Consulting all algorithms…"):
                st.session_state.compare_results = {
                    "📐 KNN":          (knn.recommend(cmp_m, movies, n=top_n, exclude_ids=excl), "similarity"),
                    "🎭 Content-Based":(cbf.recommend(cmp_m, n=top_n, exclude_ids=excl),          "genre_similarity"),
                    "🔀 Hybrid":       (hyb.by_movie(cmp_m, movies, n=top_n, exclude_ids=excl),   "hybrid_score"),
                }
    else:
        cmp_u = st.selectbox("User:", all_users, key="cmp_u")
        top_n = st.slider("Top N", 3, 8, 5, key="cmp_nu")
        if st.button("⚡ Run All Algorithms", type="primary", use_container_width=True):
            with st.spinner("Consulting all algorithms…"):
                st.session_state.compare_results = {
                    "🤝 Collaborative":(cf.recommend(cmp_u, movies, n=top_n, exclude_ids=excl),  "score"),
                    "🔮 SVD":          (svd.recommend(cmp_u, movies, n=top_n, exclude_ids=excl), "predicted_rating"),
                    "🔀 Hybrid":       (hyb.for_user(cmp_u, movies, n=top_n, exclude_ids=excl),  "hybrid_score"),
                }

    if st.session_state.compare_results:
        cols     = st.columns(len(st.session_state.compare_results))
        all_sets = []
        for col, (name, (res, sc)) in zip(cols, st.session_state.compare_results.items()):
            all_sets.append(set(res["movieId"].tolist()) if not res.empty else set())
            with col:
                st.markdown(
                    f'<div style="background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.07);'
                    f'border-radius:13px;padding:14px;min-height:280px">'
                    f'<div style="font-family:\'Bebas Neue\',sans-serif;font-size:1rem;'
                    f'letter-spacing:1.5px;color:#ffd166;margin-bottom:10px">{name}</div>',
                    unsafe_allow_html=True)
                if res.empty:
                    st.write("*No results*")
                else:
                    for j, (_, row) in enumerate(res.head(top_n).iterrows()):
                        v   = row.get(sc, 0)
                        p   = GENRE_PALETTE.get(row.get("genres","").split("|")[0],{"ac":"#aaa"})
                        st.markdown(
                            f'<div style="display:flex;align-items:center;gap:8px;padding:6px 0;'
                            f'border-bottom:1px solid rgba(255,255,255,.04)">'
                            f'<span style="font-family:monospace;color:rgba(255,255,255,.18);'
                            f'min-width:20px">#{j+1}</span>'
                            f'<div style="flex:1"><div style="font-size:.78rem;font-weight:600;'
                            f'color:#eaeaf5">{row["title"][:28]}</div>'
                            f'<div style="font-size:.62rem;color:rgba(234,234,245,.3)">'
                            f'{(row.get("genres") or "")[:28]}</div></div>'
                            f'<span style="font-family:monospace;font-size:.66rem;color:{p["ac"]}">'
                            f'{v:.2f}</span></div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        # Consensus detection
        if len(all_sets) >= 2:
            consensus = all_sets[0].copy()
            for s in all_sets[1:]: consensus &= s
            if consensus:
                names = movies[movies["movieId"].isin(consensus)]["title"].tolist()
                st.success(f"🎯 **Consensus:** {', '.join(names[:5])}")
            else:
                st.info("🤔 No overlap — each algorithm sees the data differently!")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: WATCHLIST
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📋 Watchlist":
    sh("📋", "My Watchlist")
    wl = st.session_state.watchlist
    if not wl:
        st.markdown('<div style="text-align:center;padding:3rem;color:rgba(234,234,245,.22)">'
                    '<div style="font-size:3.5rem">🎞️</div>'
                    '<div style="font-size:1rem;margin-top:.7rem">Empty watchlist.</div>'
                    '<div style="font-size:.8rem;margin-top:.3rem;color:rgba(234,234,245,.15)">'
                    'Hit + Save on any movie to add it here.</div></div>',
                    unsafe_allow_html=True)
    else:
        wl_movies = movies[movies["movieId"].isin(wl)].copy()
        wl_movies["reason"] = "Saved to watchlist"

        c1, c2 = st.columns([3, 2])
        with c1:
            sh("🎞️", f"{len(wl)} Movie{'s' if len(wl)>1 else ''}")
            render_list(wl_movies, show_actions=True, extra_key="wl_list")

            # Stats
            genres_all_wl = {}
            for gs in wl_movies["genres"].dropna():
                for g in gs.split("|"):
                    genres_all_wl[g] = genres_all_wl.get(g, 0) + 1
            avg_imdb = wl_movies["imdb_rating"].dropna().mean()
            st.markdown(
                f'<div style="background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.07);'
                f'border-radius:11px;padding:12px 16px;margin-top:8px;font-size:.78rem;'
                f'color:rgba(234,234,245,.5)">'
                f'Avg IMDb: <b style="color:#ffd166">{avg_imdb:.1f}</b> · '
                f'Top genre: <b style="color:#f72585">'
                f'{max(genres_all_wl,key=genres_all_wl.get) if genres_all_wl else "—"}</b>'
                f'</div>', unsafe_allow_html=True)

            # Export
            buf = io.StringIO()
            wl_movies[["title","genres","year","director","imdb_rating"]].to_csv(buf, index=False)
            st.download_button("⬇️ Export as CSV", buf.getvalue(),
                               "my_watchlist.csv", "text/csv", use_container_width=True)
            if st.button("🗑️ Clear All", type="secondary", use_container_width=True):
                st.session_state.watchlist = []; st.rerun()

        with c2:
            sh("💡", "More Like These")
            if wl:
                more = hyb.by_movie(wl[0], movies, n=n_recs, exclude_ids=excl + wl)
                render_list(more, "hybrid_score", show_actions=True, extra_key="wl_more")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊 Analytics":
    sh("📊", "Analytics")

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1: st.metric("Movies",       n_m)
    with m2: st.metric("Users",        n_u)
    with m3: st.metric("Ratings",      f"{n_r:,}")
    with m4: st.metric("Avg ★",        f"{ratings['rating'].mean():.2f}")
    with m5: st.metric("Avg/User",     f"{n_r/n_u:.0f}")

    t1, t2, t3, t4 = st.tabs(["🎭 Genres", "📈 Distributions", "🎬 Directors", "🏆 Evaluation"])

    with t1:
        gc = {}
        for gs in movies["genres"].dropna():
            for g in gs.split("|"): gc[g] = gc.get(g, 0) + 1
        gdf = pd.DataFrame(list(gc.items()), columns=["Genre","Count"]).sort_values("Count", ascending=False)
        st.bar_chart(gdf.set_index("Genre"), use_container_width=True, height=260)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Most-Rated Movies")
            mr = (avg_ratings.merge(movies[["movieId","title"]], on="movieId")
                  .nlargest(10,"cnt").rename(columns={"cnt":"Ratings","avg":"Avg ★"}))
            st.dataframe(mr[["title","Ratings","Avg ★"]].round(2),
                         use_container_width=True, hide_index=True)
        with c2:
            st.markdown("#### Highest Rated (≥8 votes)")
            hr = (avg_ratings.query("cnt>=8").merge(movies[["movieId","title","imdb_rating"]], on="movieId")
                  .nlargest(10,"avg").rename(columns={"avg":"Community ★"}))
            st.dataframe(hr[["title","Community ★","imdb_rating"]].round(2),
                         use_container_width=True, hide_index=True)

    with t2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Rating Distribution")
            st.bar_chart(ratings["rating"].value_counts().sort_index(),
                         use_container_width=True, height=220)
        with c2:
            st.markdown("#### Films by Decade")
            movies["decade"] = (movies["year"].fillna(0).astype(int) // 10 * 10).astype(str) + "s"
            dd = movies["decade"].value_counts().sort_index()
            st.bar_chart(dd, use_container_width=True, height=220)

        st.markdown("#### Language Distribution")
        if "language" in movies.columns:
            lc = movies["language"].value_counts().head(10)
            st.bar_chart(lc, use_container_width=True, height=180)

    with t3:
        sh("🎬", "Director Leaderboard")
        dir_stats = (movies.groupby("director")
                     .agg(films=("movieId","count"), avg_imdb=("imdb_rating","mean"))
                     .reset_index().sort_values("films", ascending=False).head(15))
        st.dataframe(dir_stats.round(2), use_container_width=True, hide_index=True)

        sel_dir = st.selectbox("Explore a director's filmography:",
                               all_directors, key="dir_explore")
        dir_films = (movies[movies["director"].str.contains(sel_dir, na=False, case=False)]
                     .merge(avg_ratings, on="movieId", how="left")
                     .sort_values("year", ascending=True))
        if not dir_films.empty:
            for _, r in dir_films.iterrows():
                yr = str(int(float(r["year"]))) if pd.notna(r.get("year")) else "—"
                avg_s = f"{r['avg']:.1f} ★" if pd.notna(r.get("avg")) else "—"
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:12px;padding:7px 0;'
                    f'border-bottom:1px solid rgba(255,255,255,.04)">'
                    f'<span style="font-family:monospace;font-size:.72rem;color:rgba(234,234,245,.3);min-width:36px">{yr}</span>'
                    f'<div style="flex:1"><span style="font-weight:600;font-size:.88rem">{r["title"]}</span> '
                    f'{genre_tags_html(r.get("genres",""), 2)}</div>'
                    f'<span style="font-family:monospace;font-size:.72rem;color:#ffd166">'
                    f'⭐ {r.get("imdb_rating","—")} · 👥 {avg_s}</span></div>',
                    unsafe_allow_html=True)

    with t4:
        st.markdown("#### Precision@K · Recall@K · NDCG@K")
        st.markdown('<p style="color:rgba(234,234,245,.42);font-size:.82rem">'
                    'Evaluates models on a 20% held-out test set for 50 sampled users.</p>',
                    unsafe_allow_html=True)
        if st.button("▶️ Run Evaluation", type="primary", use_container_width=True):
            with st.spinner("Evaluating…"):
                ev = run_evaluation(ratings, movies, cf, svd, sample_n=50)
            if not ev.empty:
                st.dataframe(ev, use_container_width=True)
                st.bar_chart(ev, use_container_width=True, height=240)
                st.success("✅ Evaluation complete!")
            else:
                st.warning("Not enough overlapping data.")
