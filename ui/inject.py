"""Inject Streamlit global theme styles from ui/*.css."""

from pathlib import Path

import streamlit as st

_UI_DIR = Path(__file__).resolve().parent


def _read_css(name: str) -> str:
    return (_UI_DIR / name).read_text(encoding="utf-8")


def inject_theme() -> None:
    """Load tokens + theme into the app once per run (Streamlit re-executes script)."""
    combined = "\n".join(
        [
            _read_css("tokens.css"),
            _read_css("theme.css"),
            _read_css("refresh.css"),
            _read_css("opening.css"),
            _read_css("report.css"),
        ]
    )
    st.markdown(f"<style>\n{combined}\n</style>", unsafe_allow_html=True)
