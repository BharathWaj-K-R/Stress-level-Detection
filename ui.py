from __future__ import annotations

import base64
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
HERO_PATH = BASE_DIR / "assets" / "hero-illustration.svg"
FAVICON_PATH = BASE_DIR / "assets" / "favicon.svg"


def _svg_data_uri(path: Path) -> str:
    try:
        raw = path.read_bytes()
    except OSError:
        return ""
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


THEME_CSS = r'''
:root {
    --bg: #f8fafc;
    --surface: #ffffff;
    --surface-soft: #f4f5f7;
    --surface-muted: #eef0f2;
    --fg: #111315;
    --muted: #6d737b;
    --border: #dfe3e7;
    --border-strong: #c9ced4;
    --black: #090a0b;
    --white: #ffffff;
    --shadow-xs: 0 1px 2px rgba(0,0,0,.03);
    --shadow-sm: 0 5px 18px rgba(0,0,0,.045);
    --shadow-md: 0 14px 34px rgba(0,0,0,.075);
    --radius-lg: 18px;
    --radius-md: 13px;
    --radius-sm: 10px;
}

.stApp {
    min-height: 100vh;
    background:
        radial-gradient(circle at 85% 0%, rgba(0,0,0,.025), transparent 24rem),
        radial-gradient(circle at 6% 40%, rgba(0,0,0,.018), transparent 20rem),
        var(--bg);
    color: var(--fg);
}

[data-testid="stHeader"] {
    background: rgba(248,250,252,.86);
    backdrop-filter: blur(12px);
}

[data-testid="stMainBlockContainer"] {
    max-width: 1320px;
    padding-top: 1.4rem;
    padding-bottom: 3.2rem;
}

[data-testid="stSidebar"] {
    background: rgba(250,251,252,.98);
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: .48rem; }
[data-testid="stSidebar"] * { color: var(--fg); }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { margin-bottom: 0; }

/* Native Streamlit container -> shadcn-like Card */
[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    background: rgba(255,255,255,.98) !important;
    box-shadow: var(--shadow-xs) !important;
}

/* Sidebar navigation as button-like pills */
[data-testid="stSidebar"] [role="radiogroup"] { gap: .25rem; }
[data-testid="stSidebar"] [role="radiogroup"] > label {
    margin: 0 !important;
    padding: .62rem .72rem;
    border: 1px solid transparent;
    border-radius: 10px;
    font-weight: 650;
    transition: all .14s ease;
}
[data-testid="stSidebar"] [role="radiogroup"] > label:hover {
    background: var(--surface-soft);
    border-color: var(--border);
}
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {
    background: var(--black);
    border-color: var(--black);
    color: #fff !important;
    box-shadow: 0 7px 16px rgba(0,0,0,.10);
}
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) * { color: #fff !important; }

/* Buttons */
.stButton > button,
.stDownloadButton > button {
    min-height: 2.62rem;
    border: 1px solid var(--border-strong);
    border-radius: 10px;
    background: var(--surface);
    color: var(--fg);
    font-weight: 700;
    box-shadow: var(--shadow-xs);
    transition: transform .14s ease, border-color .14s ease, background .14s ease;
}
.stButton > button:hover,
.stDownloadButton > button:hover {
    border-color: #aeb4bb;
    background: #f2f3f5;
    color: var(--fg);
    transform: translateY(-1px);
}
.stButton > button[kind="primary"] {
    background: var(--black) !important;
    border-color: var(--black) !important;
    color: var(--white) !important;
    box-shadow: 0 10px 22px rgba(0,0,0,.12);
}
.stButton > button[kind="primary"]:hover {
    background: #292b2d !important;
    border-color: #292b2d !important;
}

/* Inputs */
[data-baseweb="input"] > div,
[data-baseweb="textarea"] > div,
[data-baseweb="select"] > div {
    border-color: var(--border-strong) !important;
    border-radius: 10px !important;
    background: #fff !important;
}
[data-baseweb="input"] > div:focus-within,
[data-baseweb="textarea"] > div:focus-within,
[data-baseweb="select"] > div:focus-within {
    border-color: var(--fg) !important;
    box-shadow: 0 0 0 1px var(--fg) !important;
}

/* File uploader */
[data-testid="stFileUploader"] section {
    min-height: 190px;
    border: 1px dashed #bcc2c9;
    border-radius: 13px;
    background: linear-gradient(180deg, #ffffff 0%, #fafbfc 100%);
    transition: all .15s ease;
}
[data-testid="stFileUploader"] section:hover {
    border-color: #848a91;
    background: #ffffff;
    box-shadow: inset 0 0 0 1px rgba(0,0,0,.02);
}

/* Native widgets */
[data-testid="stAlert"] { border: 1px solid var(--border); border-radius: 11px; box-shadow: none; }
[data-testid="stExpander"] { border: 1px solid var(--border) !important; border-radius: 11px !important; background:#fff; }
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 11px; overflow:hidden; }

/* Brand */
.ui-brand { display:flex; align-items:center; gap:.8rem; padding:.25rem 0 .55rem; }
.ui-brand-icon { width:48px; height:48px; flex:0 0 48px; display:block; }
.ui-brand-name { font-size:1.02rem; line-height:1.04; font-weight:850; letter-spacing:-.035em; }
.ui-brand-sub { margin-top:.28rem; color:var(--muted); font-size:.72rem; }

/* Hero */
.ui-hero {
    position:relative;
    overflow:hidden;
    min-height:265px;
    padding:2.05rem 2.15rem 1.65rem;
    border:1px solid var(--border);
    border-radius:var(--radius-lg);
    background:#fff;
    box-shadow:var(--shadow-sm);
    margin-bottom:1.2rem;
}
.ui-hero-art { position:absolute; width:52%; max-width:650px; min-width:420px; right:-1%; top:-1%; bottom:0; opacity:.98; pointer-events:none; object-fit:cover; object-position:center; }
.ui-hero-copy { position:relative; z-index:2; max-width:63%; }
.ui-eyebrow { display:inline-flex; align-items:center; gap:.42rem; padding:.42rem .66rem; border:1px solid var(--border); border-radius:999px; background:#fbfbfa; color:#4e545b; font-size:.66rem; line-height:1; font-weight:800; letter-spacing:.12em; text-transform:uppercase; }
.ui-eyebrow-dot { width:6px; height:6px; border-radius:999px; background:#141516; }
.ui-title { margin:.7rem 0 0; font-size:clamp(2.45rem,5vw,4.55rem); line-height:.96; letter-spacing:-.075em; font-weight:920; color:var(--fg); }
.ui-subtitle { margin-top:.86rem; max-width:700px; color:#50565e; font-size:.98rem; line-height:1.62; }
.ui-pills { display:flex; flex-wrap:wrap; gap:.5rem; margin-top:1.05rem; }
.ui-pill { display:inline-flex; align-items:center; gap:.4rem; padding:.5rem .68rem; border:1px solid var(--border); border-radius:999px; background:#fff; color:#5d636a; font-size:.69rem; font-weight:680; box-shadow:var(--shadow-xs); }
.ui-pill-dot { width:6px; height:6px; border-radius:999px; background:#17191b; }

/* Section */
.ui-section-title { font-size:1.26rem; line-height:1.15; font-weight:850; letter-spacing:-.035em; }
.ui-section-desc { margin-top:.35rem; color:var(--muted); font-size:.8rem; line-height:1.5; }

/* Metrics */
.ui-metric { min-height:88px; padding:.88rem .95rem; border:1px solid var(--border); border-radius:11px; background:#fff; box-shadow:var(--shadow-xs); }
.ui-metric-label { color:var(--muted); font-size:.61rem; line-height:1.3; font-weight:780; letter-spacing:.09em; text-transform:uppercase; }
.ui-metric-value { margin-top:.34rem; color:var(--fg); font-size:1.34rem; line-height:1.06; font-weight:850; letter-spacing:-.045em; }

/* Empty result */
.ui-empty { min-height:238px; display:flex; flex-direction:column; align-items:center; justify-content:center; padding:1.25rem; border:1px dashed #d4d8dc; border-radius:13px; background:linear-gradient(180deg,#fff 0%,#fbfcfd 100%); text-align:center; }
.ui-empty-icon { width:58px; height:58px; border:1px solid var(--border); border-radius:999px; display:grid; place-items:center; color:#222528; background:#fff; box-shadow:var(--shadow-xs); }
.ui-empty-title { margin-top:.75rem; font-size:1rem; font-weight:820; letter-spacing:-.02em; }
.ui-empty-desc { margin-top:.38rem; max-width:300px; color:var(--muted); font-size:.78rem; line-height:1.55; }

/* Result */
.ui-result { position:relative; overflow:hidden; padding:1.5rem; min-height:245px; border-radius:16px; background:var(--black); box-shadow:var(--shadow-md); color:#fff; }
.ui-result::before,.ui-result::after { content:""; position:absolute; border:1px solid rgba(255,255,255,.08); border-radius:50%; pointer-events:none; }
.ui-result::before { width:260px; height:260px; right:-115px; top:-145px; }
.ui-result::after { width:175px; height:175px; right:-65px; top:-100px; }
.ui-result-label { position:relative; color:#a8abae; font-size:.63rem; font-weight:780; letter-spacing:.15em; text-transform:uppercase; }
.ui-result-value { position:relative; margin-top:.42rem; color:#fff; font-size:clamp(2.35rem,4.3vw,3.55rem); line-height:1; font-weight:920; letter-spacing:-.065em; }
.ui-result-meta { position:relative; max-width:560px; margin-top:.8rem; color:#d1d4d7; font-size:.79rem; line-height:1.62; }

/* Steps */
.ui-step { display:flex; gap:.85rem; padding:.9rem 0; border-bottom:1px solid var(--border); }
.ui-step:last-child { border-bottom:0; }
.ui-step-num { flex:0 0 29px; width:29px; height:29px; display:grid; place-items:center; border-radius:999px; border:1px solid var(--border); background:#fff; font-size:.68rem; font-weight:820; }
.ui-muted { color:var(--muted); font-size:.79rem; line-height:1.55; }

/* Footer */
.ui-footer { margin-top:2.6rem; padding-top:1rem; border-top:1px solid var(--border); color:#7a7f85; font-size:.68rem; line-height:1.6; text-align:center; }

@media (max-width: 900px) {
    .ui-hero { min-height:310px; }
    .ui-hero-copy { max-width:100%; }
    .ui-hero-art { width:78%; min-width:320px; opacity:.24; right:-18%; top:22%; }
}
@media (max-width: 760px) {
    [data-testid="stMainBlockContainer"] { padding-top:1rem; }
    .ui-hero { padding:1.15rem; border-radius:13px; }
    .ui-title { font-size:2.35rem; }
    .ui-pills { gap:.35rem; }
    .ui-pill { padding:.43rem .58rem; font-size:.64rem; }
}
'''


def apply_theme() -> None:
    st.markdown(f"<style>{THEME_CSS}</style>", unsafe_allow_html=True)


def brand_block(name: str = "Stress Classification Lab", subtitle: str = "Handwriting Analysis") -> None:
    icon = _svg_data_uri(FAVICON_PATH)
    if icon:
        icon_html = f'<img class="ui-brand-icon" src="{icon}" alt="" />'
    else:
        icon_html = '<div class="ui-brand-icon"></div>'
    st.markdown(
        f'''<div class="ui-brand">{icon_html}<div><div class="ui-brand-name">{name}</div><div class="ui-brand-sub">{subtitle}</div></div></div>''',
        unsafe_allow_html=True,
    )


def hero_header(title: str, subtitle: str, status: str, feature_pills: list[str]) -> None:
    art = _svg_data_uri(HERO_PATH)
    art_html = f'<img class="ui-hero-art" src="{art}" alt="" />' if art else ""
    pills = "".join(
        f'<span class="ui-pill"><span class="ui-pill-dot"></span>{item}</span>' for item in feature_pills
    )
    st.markdown(
        f'''<section class="ui-hero">
            {art_html}
            <div class="ui-hero-copy">
                <div class="ui-eyebrow"><span class="ui-eyebrow-dot"></span>{status}</div>
                <div class="ui-title">{title}</div>
                <div class="ui-subtitle">{subtitle}</div>
                <div class="ui-pills">{pills}</div>
            </div>
        </section>''',
        unsafe_allow_html=True,
    )


def section_heading(title: str, description: str | None = None) -> None:
    st.markdown(f'<div class="ui-section-title">{title}</div>', unsafe_allow_html=True)
    if description:
        st.markdown(f'<div class="ui-section-desc">{description}</div>', unsafe_allow_html=True)
        

def metric_card(label: str, value: str | int | float) -> None:
    st.markdown(
        f'''<div class="ui-metric"><div class="ui-metric-label">{label}</div><div class="ui-metric-value">{value}</div></div>''',
        unsafe_allow_html=True,
    )


def result_card(label: str, value: str, meta: str = "") -> None:
    st.markdown(
        f'''<div class="ui-result"><div class="ui-result-label">{label}</div><div class="ui-result-value">{value}</div><div class="ui-result-meta">{meta}</div></div>''',
        unsafe_allow_html=True,
    )


def empty_result_card(title: str, description: str) -> None:
    st.markdown(
        f'''<div class="ui-empty"><div class="ui-empty-icon"><svg width="25" height="25" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><path d="M12 3v18M3 12h18"/><circle cx="12" cy="12" r="8.5"/></svg></div><div class="ui-empty-title">{title}</div><div class="ui-empty-desc">{description}</div></div>''',
        unsafe_allow_html=True,
    )


def step_card(number: int, title: str, description: str) -> None:
    st.markdown(
        f'''<div class="ui-step"><div class="ui-step-num">{number}</div><div><strong>{title}</strong><br><span class="ui-muted">{description}</span></div></div>''',
        unsafe_allow_html=True,
    )


@contextmanager
def card() -> Iterator[None]:
    with st.container(border=True):
        yield


def footer(text: str) -> None:
    st.markdown(f'<div class="ui-footer">{text}</div>', unsafe_allow_html=True)
