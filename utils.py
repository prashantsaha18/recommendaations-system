"""utils.py — CSS, SVG poster engine, render helpers for CineMatch v4"""
import streamlit as st
import pandas as pd
import numpy as np
import hashlib

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN SYSTEM
# ─────────────────────────────────────────────────────────────────────────────
GENRE_PALETTE = {
    "Action":    {"fg":"#ff4d6d","bg":"#1a020a","ac":"#ff6b35","icon":"⚡"},
    "Adventure": {"fg":"#ff9f1c","bg":"#1a0e00","ac":"#ffbf69","icon":"🗺"},
    "Animation": {"fg":"#06d6a0","bg":"#00120d","ac":"#40e0bc","icon":"🎨"},
    "Comedy":    {"fg":"#ffd166","bg":"#1a1400","ac":"#ffdf8a","icon":"😄"},
    "Crime":     {"fg":"#90caf9","bg":"#050e18","ac":"#64b5f6","icon":"🔍"},
    "Drama":     {"fg":"#b39ddb","bg":"#0d0820","ac":"#9575cd","icon":"🎭"},
    "Fantasy":   {"fg":"#ce93d8","bg":"#120018","ac":"#ba68c8","icon":"🔮"},
    "Horror":    {"fg":"#ef5350","bg":"#150000","ac":"#e53935","icon":"💀"},
    "Music":     {"fg":"#4dd0e1","bg":"#001619","ac":"#00acc1","icon":"🎵"},
    "Mystery":   {"fg":"#4fc3f7","bg":"#021520","ac":"#0288d1","icon":"👁"},
    "Romance":   {"fg":"#f48fb1","bg":"#1a0510","ac":"#e91e63","icon":"❤"},
    "Sci-Fi":    {"fg":"#00e5ff","bg":"#001a21","ac":"#00b8d4","icon":"🚀"},
    "Thriller":  {"fg":"#ff7043","bg":"#1a0800","ac":"#f4511e","icon":"⚠"},
    "Biography": {"fg":"#bcaaa4","bg":"#120e0d","ac":"#a1887f","icon":"📖"},
    "History":   {"fg":"#c9a96e","bg":"#140e00","ac":"#a07850","icon":"🏛"},
    "Family":    {"fg":"#80deea","bg":"#031418","ac":"#4dd0e1","icon":"🌟"},
    "War":       {"fg":"#90a4ae","bg":"#0d1214","ac":"#607d8b","icon":"🎖"},
    "Sport":     {"fg":"#aed581","bg":"#0d1205","ac":"#7cb342","icon":"🏆"},
}

MOOD_PALETTE = {
    "😤 Adrenaline Rush":   "#ff4d6d",
    "🧠 Mind-Bending":      "#9b8ef0",
    "😢 Emotional Journey": "#4fc3f7",
    "😂 Something Fun":     "#ffd166",
    "🌑 Dark & Gritty":     "#607d8b",
    "✨ Epic & Grand":      "#ffb74d",
    "🌫️ Atmospheric":      "#80cbc4",
    "🕵️ Mystery & Crime":  "#90caf9",
    "❤️ Romance":           "#f48fb1",
    "🤯 WTF Cinema":        "#ce93d8",
}

def _genre_info(genres_str: str) -> dict:
    parts = (genres_str or "").split("|")
    for g in parts:
        if g in GENRE_PALETTE:
            return {**GENRE_PALETTE[g], "name": g}
    return {"fg":"#aaa","bg":"#111","ac":"#888","icon":"🎬","name":"Other"}

def _hash_int(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % 1000

# ─────────────────────────────────────────────────────────────────────────────
# SVG POSTER ENGINE v2
# ─────────────────────────────────────────────────────────────────────────────
def _wrap_text(text, max_chars=22) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        test = (cur + " " + w).strip()
        if len(test) > max_chars and cur:
            lines.append(cur); cur = w
        else:
            cur = test
    if cur: lines.append(cur)
    return lines[-3:]

def poster_svg(title: str, genres: str, year="", imdb="", runtime="") -> str:
    info = _genre_info(genres)
    h    = _hash_int(title) % 360
    h2   = (h + 137) % 360  # golden angle offset
    h3   = (h + 241) % 360
    fg, bg, ac, icon = info["fg"], info["bg"], info["ac"], info["icon"]
    dom  = genres.split("|")[0] if genres else "Film"
    yr   = str(int(float(year))) if year and str(year) not in ("nan","") else ""
    rt   = f"{runtime}m" if runtime and str(runtime) not in ("nan","") else ""
    imdb_s = f"★ {imdb}" if imdb and str(imdb) not in ("nan","") else ""

    # Film perforation holes
    holes = "".join(
        f'<rect x="4" y="{12+i*28}" width="10" height="18" rx="3" fill="rgba(0,0,0,.6)"/>'
        f'<rect x="186" y="{12+i*28}" width="10" height="18" rx="3" fill="rgba(0,0,0,.6)"/>'
        for i in range(10)
    )

    # Title text lines
    lines = _wrap_text(title[:30])
    title_svgs = "".join(
        f'<text x="20" y="{260 - (len(lines)-1-i)*17}" font-size="13" '
        f'font-family="Arial Black,sans-serif" font-weight="900" fill="white" '
        f'paint-order="stroke" stroke="rgba(0,0,0,.8)" stroke-width="3">{l}</text>'
        for i, l in enumerate(lines)
    )

    uid = title[:6].replace(" ","_")
    return f"""<svg viewBox="0 0 200 300" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="bg_{uid}" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%"  stop-color="hsl({h},60%,10%)"/>
      <stop offset="50%" stop-color="hsl({h2},45%,8%)"/>
      <stop offset="100%" stop-color="hsl({h3},55%,6%)"/>
    </linearGradient>
    <radialGradient id="glow_{uid}" cx="50%" cy="38%" r="52%">
      <stop offset="0%" stop-color="{ac}" stop-opacity=".22"/>
      <stop offset="70%" stop-color="{ac}" stop-opacity=".06"/>
      <stop offset="100%" stop-color="transparent"/>
    </radialGradient>
    <linearGradient id="overlay_{uid}" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%"   stop-color="rgba(0,0,0,0)"/>
      <stop offset="55%"  stop-color="rgba(0,0,0,.35)"/>
      <stop offset="100%" stop-color="rgba(0,0,0,.95)"/>
    </linearGradient>
    <filter id="grain_{uid}">
      <feTurbulence type="fractalNoise" baseFrequency="0.65" numOctaves="3" stitchTiles="stitch"/>
      <feColorMatrix type="saturate" values="0"/>
      <feBlend in="SourceGraphic" mode="multiply" result="blend"/>
      <feComposite in="blend" in2="SourceGraphic" operator="in"/>
    </filter>
  </defs>

  <!-- Base + glow -->
  <rect width="200" height="300" fill="url(#bg_{uid})"/>
  <rect width="200" height="300" fill="url(#glow_{uid})"/>

  <!-- Decorative geometric lines -->
  <line x1="20" y1="0" x2="20" y2="300" stroke="{ac}" stroke-width=".5" opacity=".15"/>
  <line x1="180" y1="0" x2="180" y2="300" stroke="{ac}" stroke-width=".5" opacity=".15"/>
  <circle cx="100" cy="108" r="52" fill="none" stroke="{ac}" stroke-width=".8" opacity=".14"/>
  <circle cx="100" cy="108" r="34" fill="none" stroke="{ac}" stroke-width=".4" opacity=".1"/>
  <line x1="20" y1="0" x2="180" y2="300" stroke="{ac}" stroke-width=".25" opacity=".06"/>
  <line x1="180" y1="0" x2="20" y2="300" stroke="{ac}" stroke-width=".25" opacity=".06"/>

  <!-- Film perforations -->
  {holes}

  <!-- Centre icon -->
  <text x="100" y="125" text-anchor="middle" dominant-baseline="middle"
    font-size="44" opacity=".18">{icon}</text>

  <!-- Film grain overlay -->
  <rect width="200" height="300" fill="rgba(128,128,128,.04)" filter="url(#grain_{uid})"/>

  <!-- Bottom overlay -->
  <rect width="200" height="300" fill="url(#overlay_{uid})"/>

  <!-- Genre badge -->
  <rect x="20" y="14" rx="8" ry="8" width="{len(dom)*7+10}" height="16"
    fill="{ac}" opacity=".2"/>
  <rect x="20" y="14" rx="8" ry="8" width="{len(dom)*7+10}" height="16"
    fill="none" stroke="{fg}" stroke-width=".8" opacity=".4"/>
  <text x="25" y="25" font-size="9" font-family="Arial,sans-serif"
    font-weight="700" fill="{fg}" letter-spacing="1">{dom.upper()}</text>

  <!-- Title -->
  {title_svgs}

  <!-- Meta row -->
  <text x="20" y="283" font-size="9" font-family="monospace" fill="rgba(255,255,255,.4)">{yr}{' · ' + rt if rt else ''}</text>
  {'<text x="180" y="283" text-anchor="end" font-size="9" font-family="monospace" fill="#ffd166">' + imdb_s + '</text>' if imdb_s else ''}
</svg>"""

# ─────────────────────────────────────────────────────────────────────────────
# HTML HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def genre_tags_html(genres_str: str, max_tags=3) -> str:
    tags = []
    for g in (genres_str or "").split("|")[:max_tags]:
        p = GENRE_PALETTE.get(g, {"fg":"#aaa","bg":"#222","ac":"#888"})
        tags.append(
            f'<span style="display:inline-block;padding:2px 8px;border-radius:999px;'
            f'margin:0 3px 3px 0;font-size:.62rem;font-weight:700;letter-spacing:.4px;'
            f'text-transform:uppercase;background:{p["bg"]};color:{p["fg"]};'
            f'border:1px solid {p["ac"]}44">{g}</span>')
    return "".join(tags)

def score_bar_html(val: float, max_val: float, color="#f72585") -> str:
    pct = min(int(val / max_val * 100), 100) if max_val else 0
    return (f'<div style="display:flex;align-items:center;gap:8px;margin-top:6px">'
            f'<div style="flex:1;height:3px;background:rgba(255,255,255,.07);border-radius:2px">'
            f'<div style="width:{pct}%;height:3px;border-radius:2px;background:{color}"></div></div>'
            f'<span style="font-family:monospace;font-size:.66rem;color:{color}">{val:.2f}</span>'
            f'</div>')

def sh(icon: str, title: str):
    """Section header."""
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:12px;margin:1.5rem 0 .8rem">'
        f'<span style="font-size:1.3rem">{icon}</span>'
        f'<span style="font-family:\'Bebas Neue\',sans-serif;font-size:1.5rem;'
        f'letter-spacing:2px;color:#eaeaf5">{title}</span>'
        f'<div style="flex:1;height:1px;background:rgba(255,255,255,.06)"></div></div>',
        unsafe_allow_html=True)

def rank_glow(i: int) -> str:
    return {0:"#ffd166",1:"#c0c0c0",2:"#cd7f32"}.get(i,"rgba(255,255,255,.08)")

# ─────────────────────────────────────────────────────────────────────────────
# CARD RENDERERS
# ─────────────────────────────────────────────────────────────────────────────
def render_poster_grid(df: pd.DataFrame, score_col=None, n=12,
                        extra_key="", show_actions=True, max_score=None):
    if df is None or df.empty:
        st.markdown('<div style="text-align:center;padding:2rem;color:rgba(234,234,245,.3)">'
                    '🎞️ No results — try adjusting your filters.</div>', unsafe_allow_html=True)
        return
    rows = [df.iloc[i:i+4] for i in range(0, min(len(df), n), 4)]
    for row_df in rows:
        cols = st.columns(len(row_df))
        for col, (i, (_, row)) in zip(cols, enumerate(row_df.iterrows())):
            mid   = int(row["movieId"])
            in_wl = mid in st.session_state.get("watchlist", [])
            svg   = poster_svg(row["title"], row.get("genres",""),
                               row.get("year",""), row.get("imdb_rating",""),
                               row.get("runtime",""))
            val_s = ""
            if score_col and score_col in df.columns:
                v = row[score_col]; val_s = f"⭐ {v:.2f}"
            with col:
                st.markdown(
                    f'<div style="border-radius:14px;overflow:hidden;position:relative;'
                    f'box-shadow:0 8px 28px rgba(0,0,0,.55);transition:transform .2s;'
                    f'margin-bottom:4px">{svg}</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div style="font-size:.72rem;font-weight:600;color:#eaeaf5;'
                    f'line-height:1.3;margin:4px 0 2px;text-align:center">{row["title"][:28]}</div>'
                    + (f'<div style="text-align:center;font-size:.63rem;color:rgba(234,234,245,.4)">'
                       f'{val_s}</div>' if val_s else ""),
                    unsafe_allow_html=True)
                if show_actions:
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("✓" if in_wl else "+", key=f"wlg_{mid}_{extra_key}_{i}",
                                     help="Watchlist"):
                            if in_wl: st.session_state.watchlist.remove(mid)
                            else:     st.session_state.watchlist.append(mid)
                            st.rerun()
                    with c2:
                        ni = mid in st.session_state.get("not_interested", [])
                        if st.button("✕", key=f"nig_{mid}_{extra_key}_{i}",
                                     help="Not interested"):
                            if ni: st.session_state.not_interested.remove(mid)
                            else:  st.session_state.not_interested.append(mid)
                            st.rerun()


def render_list(df: pd.DataFrame, score_col=None, n=12,
                extra_key="", show_actions=True, max_score=None):
    if df is None or df.empty:
        st.markdown('<div style="text-align:center;padding:2rem;color:rgba(234,234,245,.3)">'
                    '🎞️ No results.</div>', unsafe_allow_html=True)
        return
    for i, (_, row) in enumerate(df.head(n).iterrows()):
        mid   = int(row["movieId"])
        in_wl = mid in st.session_state.get("watchlist", [])
        svg   = poster_svg(row["title"], row.get("genres",""),
                           row.get("year",""), row.get("imdb_rating",""),
                           row.get("runtime",""))
        glow  = rank_glow(i)
        wl_badge = ('<span style="display:inline-block;padding:1px 8px;border-radius:999px;'
                    'font-size:.6rem;background:rgba(6,214,160,.12);border:1px solid rgba(6,214,160,.3);'
                    'color:#06d6a0;margin-left:6px">✓</span>') if in_wl else ""

        yr = str(int(float(row["year"]))) if "year" in row and str(row.get("year")) not in ("nan","") else ""
        sc_html = ""
        if score_col and score_col in df.columns:
            v  = row[score_col]
            mx = max_score or df[score_col].max() or 1
            sc_html = score_bar_html(v, mx)
        reason = str(row.get("reason",""))[:65] if row.get("reason") else ""

        st.markdown(f"""
<div style="display:flex;gap:14px;align-items:flex-start;
  background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.06);
  border-left:3px solid {glow};border-radius:14px;
  padding:14px 16px;margin-bottom:9px;position:relative;overflow:hidden;
  animation:fadeSlideIn .3s ease {i*0.04:.2f}s both">
  <div style="width:46px;height:70px;border-radius:8px;overflow:hidden;flex-shrink:0">{svg}</div>
  <div style="flex:1;min-width:0">
    <div style="font-weight:700;font-size:.95rem;color:#eaeaf5;
      white-space:nowrap;overflow:hidden;text-overflow:ellipsis">
      {row["title"]}{wl_badge}
    </div>
    <div style="font-size:.68rem;color:rgba(234,234,245,.38);
      font-family:monospace;margin:3px 0 5px">
      {'⭐ ' + str(row.get('imdb_rating','')) + '  ' if row.get('imdb_rating') and str(row.get('imdb_rating'))!='nan' else ''}
      {yr + '  ' if yr else ''}
      {(row.get('director') or '')[:24]}
    </div>
    {genre_tags_html(row.get("genres",""))}
    {sc_html}
    {'<div style="font-size:.66rem;color:rgba(255,180,80,.7);font-style:italic;margin-top:5px">💡 ' + reason + '</div>' if reason else ''}
  </div>
  <div style="font-family:monospace;font-size:1.8rem;color:rgba(255,255,255,.07);
    flex-shrink:0;line-height:1;align-self:center">#{i+1:02d}</div>
</div>""", unsafe_allow_html=True)

        if show_actions:
            c1, c2 = st.columns([1,1])
            with c1:
                if st.button("✓ Saved" if in_wl else "+ Save", key=f"wll_{mid}_{extra_key}_{i}",
                              use_container_width=True):
                    if in_wl: st.session_state.watchlist.remove(mid)
                    else:     st.session_state.watchlist.append(mid)
                    st.rerun()
            with c2:
                ni = mid in st.session_state.get("not_interested",[])
                if st.button("↩ Undo" if ni else "✕ Skip", key=f"nil_{mid}_{extra_key}_{i}",
                              use_container_width=True):
                    if ni: st.session_state.not_interested.remove(mid)
                    else:  st.session_state.not_interested.append(mid)
                    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# FULL CSS
# ─────────────────────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
@keyframes fadeSlideIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
@keyframes pulse{0%,100%{opacity:.6}50%{opacity:1}}
@keyframes gradientShift{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
*,*::before,*::after{box-sizing:border-box}
::-webkit-scrollbar{width:5px}::-webkit-scrollbar-track{background:#06060f}
::-webkit-scrollbar-thumb{background:#333;border-radius:3px}
[data-testid="stAppViewContainer"]{background:#06060f;font-family:'Outfit',sans-serif;color:#eaeaf5}
[data-testid="stAppViewContainer"]::before{content:'';position:fixed;inset:0;z-index:-1;
  background:radial-gradient(ellipse 75% 45% at 8% 18%,rgba(247,37,133,.07) 0,transparent 55%),
    radial-gradient(ellipse 55% 55% at 92% 82%,rgba(58,12,163,.09) 0,transparent 55%),
    radial-gradient(ellipse 40% 40% at 50% 50%,rgba(255,200,50,.03) 0,transparent 45%)}
[data-testid="stSidebar"]{background:rgba(4,4,12,.97)!important;
  border-right:1px solid rgba(255,255,255,.05)!important}
h1,h2,h3{font-family:'Outfit',sans-serif!important;font-weight:700}
.hero-wrap{text-align:center;padding:1.8rem 0 1rem}
.hero-logo{font-family:'Bebas Neue',sans-serif;
  font-size:clamp(3.2rem,8vw,6.5rem);letter-spacing:8px;line-height:1;
  background:linear-gradient(135deg,#f72585 0%,#ff6b35 28%,#ffd166 55%,#06d6a0 80%,#4cc9f0 100%);
  background-size:300% 300%;animation:gradientShift 5s ease infinite;
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
  filter:drop-shadow(0 0 40px rgba(247,37,133,.3))}
.hero-sub{font-size:.82rem;color:rgba(234,234,245,.38);letter-spacing:4px;text-transform:uppercase;font-weight:300;margin-top:.3rem}
.hero-bar{width:50px;height:2px;background:linear-gradient(90deg,transparent,#f72585,transparent);margin:.7rem auto}
.stat-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:7px}
.stat-box{background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.07);
  border-radius:11px;padding:12px;text-align:center}
.stat-num{font-family:'Bebas Neue',sans-serif;font-size:1.7rem;line-height:1}
.stat-lbl{font-size:.58rem;color:rgba(234,234,245,.38);text-transform:uppercase;letter-spacing:1px;margin-top:2px}
.stTabs [data-baseweb="tab-list"]{background:rgba(255,255,255,.025);border-radius:11px;padding:3px;gap:2px;
  border:1px solid rgba(255,255,255,.06)}
.stTabs [data-baseweb="tab"]{border-radius:8px!important;color:rgba(234,234,245,.4)!important;
  font-size:.8rem!important;font-weight:500!important;padding:6px 12px!important}
.stTabs [aria-selected="true"]{background:rgba(247,37,133,.14)!important;color:#f72585!important}
div[data-baseweb="select"]>div{background:rgba(255,255,255,.04)!important;
  border-color:rgba(255,255,255,.1)!important;color:#eaeaf5!important}
[data-testid="stSelectbox"] label,[data-testid="stSlider"] label,
[data-testid="stRadio"] label,[data-testid="stMultiSelect"] label{
  color:rgba(234,234,245,.6)!important;font-size:.82rem!important}
div[data-testid="stButton"]>button{border-radius:9px!important;
  border:1px solid rgba(255,255,255,.09)!important;
  background:rgba(255,255,255,.04)!important;color:#eaeaf5!important;
  font-family:'Outfit',sans-serif!important;font-size:.82rem!important;
  transition:all .18s!important}
div[data-testid="stButton"]>button:hover{
  border-color:rgba(247,37,133,.45)!important;
  background:rgba(247,37,133,.09)!important;transform:translateY(-1px)!important}
div[data-testid="stButton"]>button[kind="primary"]{
  background:linear-gradient(135deg,#f72585,#b5179e)!important;
  border:none!important;color:#fff!important;font-weight:600!important;
  box-shadow:0 4px 16px rgba(247,37,133,.25)!important}
div[data-testid="stExpander"]{background:rgba(255,255,255,.02)!important;
  border:1px solid rgba(255,255,255,.06)!important;border-radius:11px!important}
.stMetric{background:rgba(255,255,255,.03)!important;border-radius:11px!important;
  padding:14px!important;border:1px solid rgba(255,255,255,.06)!important}
.stMetric label{color:rgba(234,234,245,.4)!important;font-size:.7rem!important}
[data-testid="stMetricValue"]{color:#ffd166!important;font-family:'JetBrains Mono',monospace!important}
div[data-testid="stDownloadButton"]>button{border-color:rgba(6,214,160,.3)!important;
  color:#06d6a0!important}
div[data-testid="stDownloadButton"]>button:hover{background:rgba(6,214,160,.1)!important}
</style>
"""

def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)

def hero():
    st.markdown("""<div class="hero-wrap">
      <div class="hero-logo">CineMatch</div>
      <div class="hero-sub">Discover · Rate · Obsess</div>
      <div class="hero-bar"></div>
    </div>""", unsafe_allow_html=True)