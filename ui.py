from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import streamlit as st


THEME_CSS = r'''
:root {
    --background: #f8f8f7;
    --foreground: #161616;
    --card: #ffffff;
    --card-foreground: #161616;
    --muted: #f1f1ef;
    --muted-foreground: #70706a;
    --border: #deded9;
    --input: #d5d5d0;
    --ring: #161616;
    --accent: #f1f1ef;
    --accent-foreground: #161616;
    --radius: 0.75rem;
    --shadow-sm: 0 1px 2px rgba(0,0,0,.04);
    --shadow-md: 0 8px 24px rgba(0,0,0,.06);
}

.stApp {
    background: var(--background);
    color: var(--foreground);
}

[data-testid="stMainBlockContainer"] {
    max-width: 1280px;
    padding-top: 2.25rem;
    padding-bottom: 2.5rem;
}

[data-testid="stSidebar"] {
    background: #fbfbfa;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: .6rem;
}

[data-testid="stSidebar"] * {
    color: var(--foreground);
}

/* shadcn-like bordered container */
[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    background: var(--card) !important;
    box-shadow: var(--shadow-sm) !important;
}

[data-testid="stVerticalBlockBorderWrapper"] > div {
    border-radius: inherit;
}

/* Typography */
h1, h2, h3, h4 {
    color: var(--foreground) !important;
    letter-spacing: -.025em;
}

p, label, [data-testid="stCaptionContainer"] {
    color: var(--foreground);
}

[data-testid="stCaptionContainer"] {
    color: var(--muted-foreground) !important;
}

/* Buttons */
.stButton > button,
.stDownloadButton > button {
    min-height: 2.5rem;
    border-radius: .55rem;
    border: 1px solid var(--border);
    background: var(--card);
    color: var(--foreground);
    font-weight: 650;
    box-shadow: var(--shadow-sm);
    transition: all .14s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    border-color: #bab9b2;
    background: #f3f3f1;
    color: var(--foreground);
}

.stButton > button[kind="primary"] {
    background: var(--foreground) !important;
    border-color: var(--foreground) !important;
    color: #ffffff !important;
}

.stButton > button[kind="primary"]:hover {
    background: #353533 !important;
    border-color: #353533 !important;
}

/* Inputs */
[data-baseweb="input"] > div,
[data-baseweb="textarea"] > div,
[data-baseweb="select"] > div {
    border-color: var(--input) !important;
    border-radius: .55rem !important;
    background: #ffffff !important;
}

[data-baseweb="input"] > div:focus-within,
[data-baseweb="textarea"] > div:focus-within,
[data-baseweb="select"] > div:focus-within {
    border-color: var(--ring) !important;
    box-shadow: 0 0 0 1px var(--ring) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: .2rem;
    border-bottom: 1px solid var(--border);
}

.stTabs [data-baseweb="tab"] {
    padding: .55rem .8rem;
    color: var(--muted-foreground);
}

.stTabs [aria-selected="true"] {
    color: var(--foreground) !important;
}

/* File uploader */
[data-testid="stFileUploader"] section {
    border: 1px dashed #c6c6c0;
    border-radius: var(--radius);
    background: #fcfcfb;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: .65rem;
    border: 1px solid var(--border);
    box-shadow: none;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: .65rem;
    overflow: hidden;
}

/* Expander */
[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: .65rem !important;
    background: var(--card);
}

.ui-eyebrow {
    font-size: .68rem;
    font-weight: 720;
    letter-spacing: .16em;
    text-transform: uppercase;
    color: var(--muted-foreground);
}

.ui-title {
    margin: .28rem 0 0;
    font-size: clamp(2.1rem, 4vw, 3.45rem);
    line-height: .98;
    letter-spacing: -.055em;
    font-weight: 850;
    color: var(--foreground);
}

.ui-subtitle {
    max-width: 760px;
    margin-top: .75rem;
    color: var(--muted-foreground);
    line-height: 1.65;
    font-size: .94rem;
}

.ui-status {
    margin-top: .8rem;
    color: var(--muted-foreground);
    font-size: .76rem;
}

.ui-dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    margin-right: .45rem;
    border-radius: 999px;
    background: #161616;
}

.ui-shell {
    padding-bottom: 1.35rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.55rem;
}

.ui-card {
    padding: 1rem 1.05rem;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: var(--card);
    box-shadow: var(--shadow-sm);
}

.ui-card-title {
    font-size: .9rem;
    font-weight: 760;
    color: var(--foreground);
}

.ui-card-description {
    margin-top: .25rem;
    color: var(--muted-foreground);
    font-size: .78rem;
    line-height: 1.55;
}

.ui-metric {
    min-height: 88px;
    padding: .9rem;
    border: 1px solid var(--border);
    border-radius: .7rem;
    background: var(--card);
    box-shadow: var(--shadow-sm);
}

.ui-metric-label {
    color: var(--muted-foreground);
    font-size: .66rem;
    font-weight: 720;
    letter-spacing: .09em;
    text-transform: uppercase;
}

.ui-metric-value {
    margin-top: .28rem;
    color: var(--foreground);
    font-size: 1.3rem;
    line-height: 1.1;
    font-weight: 800;
    letter-spacing: -.03em;
}

.ui-badge {
    display: inline-flex;
    align-items: center;
    min-height: 1.45rem;
    padding: .1rem .48rem;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: var(--muted);
    color: var(--foreground);
    font-size: .68rem;
    font-weight: 720;
}

.ui-badge-dark {
    background: var(--foreground);
    color: #fff;
    border-color: var(--foreground);
}

.ui-result {
    padding: 1.45rem;
    border-radius: var(--radius);
    background: var(--foreground);
    color: #fff;
    box-shadow: var(--shadow-md);
}

.ui-result-label {
    color: #a9a9a2;
    font-size: .66rem;
    font-weight: 720;
    letter-spacing: .14em;
    text-transform: uppercase;
}

.ui-result-value {
    margin-top: .35rem;
    color: #fff;
    font-size: clamp(2.1rem, 4vw, 3.2rem);
    line-height: 1;
    font-weight: 850;
    letter-spacing: -.055em;
}

.ui-result-meta {
    margin-top: .75rem;
    color: #d7d7d2;
    font-size: .8rem;
    line-height: 1.55;
}

.ui-step {
    display: flex;
    gap: .8rem;
    padding: .85rem 0;
    border-bottom: 1px solid var(--border);
}

.ui-step:last-child { border-bottom: 0; }

.ui-step-num {
    flex: 0 0 25px;
    width: 25px;
    height: 25px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: var(--card);
    color: var(--foreground);
    font-size: .69rem;
    font-weight: 800;
}

.ui-muted {
    color: var(--muted-foreground);
}

.ui-footer {
    margin-top: 2.6rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border);
    color: #777771;
    font-size: .7rem;
    line-height: 1.55;
}

@media (max-width: 720px) {
    [data-testid="stMainBlockContainer"] { padding-top: 1.35rem; }
    .ui-title { font-size: 2.3rem; }
    .ui-card, .ui-result { padding: 1rem; }
}
'''


def apply_theme() -> None:
    st.markdown(f"<style>{THEME_CSS}</style>", unsafe_allow_html=True)


def page_header(eyebrow: str, title: str, subtitle: str, status: str = "") -> None:
    status_html = f'<div class="ui-status"><span class="ui-dot"></span>{status}</div>' if status else ""
    st.markdown(
        f'''<div class="ui-shell">
            <div class="ui-eyebrow">{eyebrow}</div>
            <div class="ui-title">{title}</div>
            <div class="ui-subtitle">{subtitle}</div>
            {status_html}
        </div>''',
        unsafe_allow_html=True,
    )


def section_heading(title: str, description: str | None = None) -> None:
    st.markdown(f"### {title}")
    if description:
        st.caption(description)


def metric_card(label: str, value: str | int | float) -> None:
    st.markdown(
        f'''<div class="ui-metric">
            <div class="ui-metric-label">{label}</div>
            <div class="ui-metric-value">{value}</div>
        </div>''',
        unsafe_allow_html=True,
    )


def badge(text: str, dark: bool = False) -> None:
    class_name = "ui-badge ui-badge-dark" if dark else "ui-badge"
    st.markdown(f'<span class="{class_name}">{text}</span>', unsafe_allow_html=True)


def result_card(label: str, value: str, meta: str = "") -> None:
    st.markdown(
        f'''<div class="ui-result">
            <div class="ui-result-label">{label}</div>
            <div class="ui-result-value">{value}</div>
            <div class="ui-result-meta">{meta}</div>
        </div>''',
        unsafe_allow_html=True,
    )


def step_card(number: int, title: str, description: str) -> None:
    st.markdown(
        f'''<div class="ui-step">
            <div class="ui-step-num">{number}</div>
            <div><strong>{title}</strong><br>
            <span class="ui-muted">{description}</span></div>
        </div>''',
        unsafe_allow_html=True,
    )


@contextmanager
def card() -> Iterator[None]:
    with st.container(border=True):
        yield


def footer(text: str) -> None:
    st.markdown(f'<div class="ui-footer">{text}</div>', unsafe_allow_html=True)
