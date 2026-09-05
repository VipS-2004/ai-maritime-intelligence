import hashlib
import re

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from models.detector import load_model, run_detection
from analysis.maritime_analysis import generate_analysis
from ai.agent import generate_intelligence_report, GeminiQuotaError
from visualization.visualizer import generate_heatmap, zone_based_analysis


PERCEPTION_IMAGE_WIDTH = 560
SPATIAL_IMAGE_WIDTH = 420
PREVIEW_IMAGE_WIDTH = 480


st.set_page_config(
    page_title="Satellite Maritime Intelligence",
    layout="wide",
    initial_sidebar_state="collapsed",
)


if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False

if "file_signature" not in st.session_state:
    st.session_state.file_signature = None

if "uploaded_name" not in st.session_state:
    st.session_state.uploaded_name = None

if "uploaded_bytes" not in st.session_state:
    st.session_state.uploaded_bytes = None

if "model" not in st.session_state:
    st.session_state.model = None

if "results" not in st.session_state:
    st.session_state.results = None

if "analysis" not in st.session_state:
    st.session_state.analysis = None

if "orig_img" not in st.session_state:
    st.session_state.orig_img = None

if "heatmap" not in st.session_state:
    st.session_state.heatmap = None

if "zone_counts" not in st.session_state:
    st.session_state.zone_counts = None

if "hotspots" not in st.session_state:
    st.session_state.hotspots = None

if "zone_overlay" not in st.session_state:
    st.session_state.zone_overlay = None

if "intelligence_report" not in st.session_state:
    st.session_state.intelligence_report = None

if "intelligence_error" not in st.session_state:
    st.session_state.intelligence_error = None

if "pipeline_stage" not in st.session_state:
    st.session_state.pipeline_stage = "mission"


PIPELINE_STAGES = [
    ("mission", "MISSION"),
    ("perception", "PERCEPTION"),
    ("analysis", "ANALYSIS"),
    ("spatial", "SPATIAL"),
    ("risk", "RISK"),
    ("ai", "AI"),
]

OPENING_PIPELINE_STAGES = [
    ("mission", "SOURCE"),
    ("perception", "DETECT"),
    ("analysis", "UNDERSTAND"),
    ("spatial", "LOCATE"),
    ("risk", "ASSESS"),
    ("ai", "INTERPRET"),
]


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

    .stApp {
        background: #060b12;
        color: #d8e0ea;
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .main .block-container,
    div.block-container,
    [data-testid="stMainBlockContainer"] {
        max-width: 1360px;
        width: 100%;
        margin-left: auto !important;
        margin-right: auto !important;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
        padding-top: 0.85rem;
        padding-bottom: 2.5rem;
    }

    div[data-testid="stVerticalBlock"] > div {
        gap: 0.45rem;
    }

    h1, h2, h3, h4 {
        color: #eef3f8 !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-weight: 600 !important;
    }

    h1 {
        font-size: 1.65rem !important;
        letter-spacing: -0.3px;
        margin-bottom: 0.1rem !important;
        margin-top: 0 !important;
    }

    h2 {
        font-size: 1.18rem !important;
        letter-spacing: -0.15px;
        margin-top: 0.1rem !important;
        margin-bottom: 0.15rem !important;
    }

    h3 {
        font-size: 1rem !important;
        margin-top: 0.15rem !important;
        margin-bottom: 0.2rem !important;
    }

    p, .stMarkdown, .stCaption {
        color: #9aabbd;
    }

    [data-testid="stCaption"] {
        margin-bottom: 0.2rem !important;
    }

    hr {
        border-color: #152232 !important;
        margin: 0.85rem 0 !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #0b1420;
        border: 1px solid #1a2a3d !important;
        border-radius: 5px;
    }

    [data-testid="stVerticalBlockBorderWrapper"] > div {
        padding-top: 0.55rem !important;
        padding-bottom: 0.55rem !important;
    }

    [data-testid="stMetric"] {
        background: #0c1522;
        border: 1px solid #1a2a3d;
        border-radius: 5px;
        padding: 8px 11px;
    }

    [data-testid="stMetricLabel"] {
        color: #7d8fa3 !important;
        font-size: 0.6rem !important;
        font-weight: 600;
        letter-spacing: 0.07em;
        text-transform: uppercase;
    }

    [data-testid="stMetricValue"] {
        color: #eef3f8 !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-weight: 500;
        font-size: 1.15rem !important;
    }

    [data-testid="stFileUploader"] {
        background: #0a121c;
        border: 1px solid #1a2a3d;
        border-radius: 5px;
        padding: 3px;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: #0d1624;
        border-color: #243447;
    }

    .stButton > button {
        min-height: 40px;
        border-radius: 4px;
        border: 1px solid #2a6f8f;
        background: #0d2430;
        color: #d7eef7;
        font-weight: 600;
        letter-spacing: 0.04em;
    }

    .stButton > button:hover {
        border-color: #3d8fad;
        background: #123040;
        color: #ffffff;
    }

    [data-testid="stImage"] {
        border-radius: 3px;
        overflow: hidden;
    }

    [data-testid="stImage"] img {
        max-height: none !important;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #1a2a3d;
        border-radius: 5px;
        overflow: hidden;
    }

    [data-testid="stAlert"] {
        border-radius: 4px;
        padding: 0.55rem 0.75rem;
    }

    .app-kicker {
        color: #6d8094;
        font-size: 0.64rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.2rem;
    }

    .app-subtitle {
        color: #8fa0b3;
        font-size: 0.84rem;
        margin-top: 0.1rem;
        margin-bottom: 0;
        line-height: 1.35;
    }

    .status-chip {
        display: inline-block;
        padding: 5px 9px;
        border: 1px solid #1f3348;
        border-radius: 3px;
        background: #0c1522;
        color: #9eb0c2;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.06em;
    }

    .status-chip-ready {
        border-color: #1f4a3a;
        color: #7dba9a;
    }

    .status-chip-run {
        border-color: #2a6f8f;
        color: #7ec8e3;
    }

    .status-chip-done {
        border-color: #1f4a3a;
        color: #7dba9a;
    }

    .pipeline-bar {
        display: flex;
        align-items: center;
        gap: 0;
        flex-wrap: wrap;
        padding: 8px 12px;
        background: #0b1420;
        border: 1px solid #1a2a3d;
        border-radius: 5px;
        margin: 0.2rem 0 0.55rem 0;
    }

    .pipeline-item {
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .pipeline-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.64rem;
        font-weight: 500;
        letter-spacing: 0.07em;
        color: #5f7388;
        text-transform: uppercase;
    }

    .pipeline-label.pending {
        color: #5f7388;
    }

    .pipeline-label.active {
        color: #7ec8e3;
    }

    .pipeline-label.done {
        color: #8fa3b8;
    }

    .pipeline-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #2a3c50;
        flex-shrink: 0;
    }

    .pipeline-dot.active {
        background: #3d9ec0;
    }

    .pipeline-dot.done {
        background: #3d7a5f;
    }

    .pipeline-sep {
        width: 14px;
        height: 1px;
        background: #243447;
        margin: 0 6px;
        flex-shrink: 0;
    }

    .stage-kicker {
        color: #6d8094;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.64rem;
        font-weight: 500;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 0.05rem;
    }

    .stage-question {
        color: #8fa0b3;
        font-size: 0.82rem;
        margin-top: 0.05rem;
        margin-bottom: 0.45rem;
        line-height: 1.35;
    }

    .mono-value {
        font-family: 'IBM Plex Mono', monospace;
        color: #e6eef6;
        font-size: 0.82rem;
    }

    .class-row {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        gap: 10px;
        padding: 4px 0;
        border-bottom: 1px solid #152232;
        font-size: 0.84rem;
    }

    .class-row:last-child {
        border-bottom: none;
    }

    .class-name {
        color: #c5d0dc;
    }

    .class-count {
        font-family: 'IBM Plex Mono', monospace;
        color: #9eb8cc;
        font-size: 0.78rem;
    }

    .obs-note {
        color: #6d8094;
        font-size: 0.72rem;
        margin-top: 0.3rem;
        line-height: 1.35;
    }

    .config-line {
        display: flex;
        justify-content: space-between;
        gap: 10px;
        padding: 4px 0;
        border-bottom: 1px solid #152232;
        font-size: 0.82rem;
    }

    .config-line:last-child {
        border-bottom: none;
    }

    .config-key {
        color: #7d8fa3;
    }

    .config-val {
        font-family: 'IBM Plex Mono', monospace;
        color: #c5d0dc;
        font-size: 0.78rem;
    }

    .produce-list {
        color: #9aabbd;
        font-size: 0.8rem;
        line-height: 1.55;
        margin: 0;
        padding-left: 1rem;
    }

    .signal-box {
        border: 1px solid #1a2a3d;
        border-radius: 5px;
        background: #0c1522;
        padding: 10px 12px;
        margin-bottom: 0.35rem;
    }

    .signal-box.ok {
        border-color: #1f4a3a;
    }

    .signal-box.warn {
        border-color: #6b5420;
    }

    .signal-box.alert {
        border-color: #6b3030;
    }

    .signal-label {
        color: #7d8fa3;
        font-size: 0.6rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }

    .signal-text {
        color: #c5d0dc;
        font-size: 0.86rem;
        line-height: 1.4;
    }

    .risk-pill {
        display: inline-block;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        letter-spacing: 0.06em;
        padding: 4px 8px;
        border-radius: 3px;
        border: 1px solid #1a2a3d;
    }

    .risk-pill.low {
        color: #7dba9a;
        border-color: #1f4a3a;
        background: #0c1a16;
    }

    .risk-pill.medium {
        color: #d4b56a;
        border-color: #6b5420;
        background: #1a160c;
    }

    .risk-pill.high {
        color: #d98989;
        border-color: #6b3030;
        background: #1a0f0f;
    }

    .ai-panel-label {
        color: #6d8094;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.62rem;
        font-weight: 500;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }

    .ai-meta {
        color: #6d8094;
        font-size: 0.72rem;
        margin-bottom: 0.45rem;
    }

    .stApp:has(.opening-workspace) .pipeline-bar {
        background: transparent;
        border: none;
        padding: 2px 0;
        margin: 0.25rem 0 1rem 0;
        flex-wrap: nowrap;
    }

    .stApp:has(.opening-workspace):has(.opening-empty) .pipeline-bar {
        margin: 0.2rem 0 1.75rem 0;
    }

    .stApp:has(.opening-workspace) .pipeline-sep {
        width: 10px;
        margin: 0 5px;
    }

    .stApp:has(.opening-workspace) .opening-intro {
        margin: 0;
        max-width: none;
    }

    .stApp:has(.opening-workspace) .opening-lead-title {
        color: #eef3f8;
        font-size: 1.22rem;
        font-weight: 600;
        letter-spacing: -0.28px;
        line-height: 1.3;
        margin: 0 0 0.7rem 0;
        max-width: 36rem;
    }

    .stApp:has(.opening-workspace) .opening-lead-copy {
        color: #8fa0b3;
        font-size: 0.84rem;
        line-height: 1.5;
        margin: 0 0 1.55rem 0;
        max-width: 34rem;
    }

    .stApp:has(.opening-workspace) .opening-steps {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 0;
        max-width: 46rem;
        margin: 0;
    }

    .stApp:has(.opening-workspace) .opening-step {
        padding-right: 1.4rem;
    }

    .stApp:has(.opening-workspace) .opening-step + .opening-step {
        padding-left: 1.4rem;
        border-left: 1px solid #1a2a3d;
    }

    .stApp:has(.opening-workspace) .opening-step-kicker {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.64rem;
        font-weight: 500;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #7ec8e3;
        margin-bottom: 0.32rem;
    }

    .stApp:has(.opening-workspace) .opening-step-text {
        color: #8fb9c9;
        font-size: 0.8rem;
        line-height: 1.4;
        margin: 0;
        max-width: 16.5rem;
    }

    .stApp:has(.has-source-image) .opening-intro,
    .stApp:has(.has-source-image) [data-testid="stElementContainer"]:has(.opening-intro) {
        display: none !important;
        height: 0 !important;
        min-height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow: hidden !important;
    }

    .stApp:has(.opening-workspace) [data-testid="stFileUploader"] {
        background: transparent;
        border: none;
        padding: 0;
        max-width: none;
        width: 100%;
        margin: 0;
    }

    .stApp:has(.opening-workspace) [data-testid="stFileUploaderDropzone"] {
        min-height: 0 !important;
        height: auto !important;
        width: 100%;
        padding: 0.75rem 1rem !important;
    }

    .stApp:has(.has-source-image) [data-testid="stFileUploaderDropzone"],
    .stApp:has(.has-source-image) [data-testid="stFileUploaderFile"] {
        display: none !important;
    }

    .stApp:has(.has-source-image) [data-testid="stFileUploader"] {
        height: 0 !important;
        max-height: 0 !important;
        min-height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow: hidden !important;
        border: none !important;
        max-width: none;
    }

    .stApp:has(.has-source-image) [data-testid="stElementContainer"]:has([data-testid="stFileUploader"]) {
        display: none !important;
        min-height: 0 !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow: hidden !important;
    }

    .has-source-image {
        display: none;
    }

    .stApp:has(.opening-workspace) [data-testid="stHorizontalBlock"] {
        align-items: start !important;
        gap: 1.5rem !important;
    }

    .stApp:has(.opening-workspace) [data-testid="stImage"] {
        width: 100%;
        max-width: 100%;
        overflow: visible;
        background: transparent;
        border: none;
    }

    .stApp:has(.opening-workspace) [data-testid="stImage"] img {
        display: block;
        width: 100% !important;
        max-width: 100% !important;
        height: auto !important;
        max-height: min(68vh, 720px) !important;
        object-fit: contain !important;
        object-position: left center;
    }

    .stApp:has(.opening-workspace):has(.has-source-image) .main .block-container,
    .stApp:has(.opening-workspace):has(.has-source-image) div.block-container,
    .stApp:has(.opening-workspace):has(.has-source-image) [data-testid="stMainBlockContainer"] {
        padding-top: 0.55rem;
        padding-bottom: 1.25rem;
    }

    .stApp:has(.opening-workspace):has(.has-source-image) div[data-testid="stVerticalBlock"] > div {
        gap: 0.28rem;
    }

    .stApp:has(.has-source-image) .pipeline-bar {
        margin: 0.15rem 0 0.55rem 0;
    }

    .stApp:has(.has-source-image) [data-testid="stHorizontalBlock"] {
        align-items: start !important;
        gap: 1.15rem !important;
        max-width: 52rem;
    }

    .stApp:has(.has-source-image) [data-testid="stHorizontalBlock"] > div:last-child {
        border-left: 1px solid #1a2a3d;
        padding-left: 1.1rem;
    }

    .stApp:has(.has-source-image) [data-testid="stImage"] {
        width: auto;
        max-width: 100%;
    }

    .stApp:has(.has-source-image) [data-testid="stImage"] img {
        display: block;
        width: auto !important;
        max-width: 100% !important;
        height: auto !important;
        max-height: min(44vh, 400px) !important;
        object-fit: contain !important;
        object-position: left top;
    }

    .stApp:has(.has-source-image) .opening-ready-note {
        margin: 0.2rem 0 0.85rem 0;
    }

    .opening-ready-note {
        color: #8fa0b3;
        font-size: 0.82rem;
        margin: 0 0 0.75rem 0;
        line-height: 1.35;
    }

    .opening-empty {
        display: none;
    }

    .stApp:has(.opening-workspace):has(.opening-empty) .main .block-container,
    .stApp:has(.opening-workspace):has(.opening-empty) div.block-container,
    .stApp:has(.opening-workspace):has(.opening-empty) [data-testid="stMainBlockContainer"] {
        padding-top: 0.7rem;
        padding-bottom: min(28vh, 12rem);
    }

    .stApp:has(.opening-workspace):has(.opening-empty) div[data-testid="stVerticalBlock"] > div {
        gap: 0.32rem;
    }

    .stApp:has(.opening-workspace):has(.opening-empty) .app-kicker {
        margin-bottom: 0.12rem;
    }

    .stApp:has(.opening-workspace):has(.opening-empty) h1 {
        margin-bottom: 0.05rem !important;
    }

    .stApp:has(.opening-workspace):has(.opening-empty) .opening-lead-title {
        margin: 0 0 0.85rem 0;
    }

    .stApp:has(.opening-workspace):has(.opening-empty) .opening-lead-copy {
        margin: 0 0 1.9rem 0;
    }

    .stApp:has(.opening-workspace):has(.opening-empty) .opening-step-kicker {
        margin-bottom: 0.42rem;
    }

    .stApp:has(.opening-workspace):has(.opening-empty) [data-testid="stFileUploader"] {
        margin-top: 1.85rem;
        padding-top: 1.35rem;
        padding-bottom: 1.35rem;
        border-top: 1px solid #1a2a3d;
        border-bottom: 1px solid #152232;
    }

    .stApp:has(.opening-workspace):has(.opening-empty) [data-testid="stFileUploaderDropzone"] {
        border: 1px solid #1a2a3d;
        background: #0b1420;
        border-radius: 5px;
        min-height: 4.6rem !important;
        padding: 1.15rem 1.15rem !important;
    }

    .stApp:has(.opening-workspace):has(.opening-empty) [data-testid="stFileUploaderDropzoneInstructions"] {
        color: #8fa0b3;
    }

    .stApp:has(.opening-workspace):has(.opening-empty) [data-testid="stFileUploaderDropzone"] small {
        color: #6d8094;
    }

    .report-workspace {
        display: none;
    }

    .stApp:has(.report-workspace) .main .block-container,
    .stApp:has(.report-workspace) div.block-container,
    .stApp:has(.report-workspace) [data-testid="stMainBlockContainer"] {
        padding-top: 3.25rem;
    }

    .stApp:has(.report-workspace) .report-kicker {
        color: #eef3f8;
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 1.18rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin: 0;
        line-height: 1.2;
    }

    .stApp:has(.report-workspace) [data-testid="stHorizontalBlock"]:has(.report-kicker) {
        align-items: center !important;
        gap: 1rem !important;
    }

    .stApp:has(.report-workspace) .report-breadcrumb {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.45rem 0.55rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.64rem;
        font-weight: 500;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin: 0.7rem 0 0.1rem 0;
    }

    .stApp:has(.report-workspace) .report-crumb {
        color: #7ec8e3;
    }

    .stApp:has(.report-workspace) .report-crumb-sep {
        color: #2a6f8f;
        font-size: 0.7rem;
        letter-spacing: 0;
    }

    .stApp:has(.report-workspace) .report-primary-stat {
        margin: 0.15rem 0 0.75rem 0;
    }

    .stApp:has(.report-workspace) .report-primary-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2.05rem;
        font-weight: 500;
        color: #eef3f8;
        letter-spacing: -0.04em;
        line-height: 1;
        margin-bottom: 0.28rem;
    }

    .stApp:has(.report-workspace) .report-primary-label {
        color: #7ec8e3;
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .stApp:has(.report-workspace) .report-supporting {
        margin-top: 0.15rem;
    }

    .stApp:has(.report-workspace) .report-supporting .config-line,
    .stApp:has(.report-workspace) .report-supporting .class-row {
        opacity: 0.88;
    }

    .stApp:has(.report-workspace) .report-takeaway {
        margin: 0.1rem 0 0.85rem 0;
    }

    .stApp:has(.report-workspace) .report-takeaway-main {
        font-size: 1.28rem;
        font-weight: 600;
        color: #eef3f8;
        letter-spacing: -0.2px;
        line-height: 1.25;
        margin-bottom: 0.35rem;
    }

    .stApp:has(.report-workspace) .report-takeaway-sub {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.86rem;
        color: #9eb8cc;
        letter-spacing: 0.02em;
        line-height: 1.45;
    }

    .stApp:has(.report-workspace) .report-spatial-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem 1.35rem;
        margin: 1.05rem 0 0.55rem 0;
    }

    .stApp:has(.report-workspace) .report-spatial-item {
        min-width: 0;
    }

    .stApp:has(.report-workspace) .report-spatial-label {
        color: #6d8094;
        font-size: 0.6rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.12rem;
    }

    .stApp:has(.report-workspace) .report-spatial-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.86rem;
        color: #c5d0dc;
    }

    .stApp:has(.report-workspace) .report-spatial-note {
        color: #6d8094;
        font-size: 0.72rem;
        margin: 0.45rem 0 0 0;
        line-height: 1.4;
    }

    .stApp:has(.report-workspace) .report-comp {
        margin: 0.15rem 0 0.35rem 0;
    }

    .stApp:has(.report-workspace) .report-comp-heading {
        color: #7d8fa3;
        font-size: 0.6rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin: 0 0 0.45rem 0;
    }

    .stApp:has(.report-workspace) .report-comp-bar {
        display: flex;
        width: 100%;
        height: 13px;
        overflow: hidden;
        border-radius: 3px;
        background: #0c1522;
        border: 1px solid #1a2a3d;
        margin: 0 0 0.7rem 0;
    }

    .stApp:has(.report-workspace) .report-comp-seg {
        min-width: 0;
        height: 100%;
    }

    .stApp:has(.report-workspace) .report-comp-seg.military {
        background: #8b4d4d;
    }

    .stApp:has(.report-workspace) .report-comp-seg.civilian {
        background: #3d7a8f;
    }

    .stApp:has(.report-workspace) .report-comp-seg.unknown {
        background: #4d5966;
    }

    .stApp:has(.report-workspace) .report-comp-legend {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem 1.5rem;
        margin: 0 0 1rem 0;
    }

    .stApp:has(.report-workspace) .report-comp-item {
        display: flex;
        align-items: baseline;
        gap: 0.45rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #8fa0b3;
    }

    .stApp:has(.report-workspace) .report-comp-dot {
        width: 7px;
        height: 7px;
        border-radius: 1px;
        flex-shrink: 0;
        position: relative;
        top: -1px;
    }

    .stApp:has(.report-workspace) .report-comp-dot.military {
        background: #8b4d4d;
    }

    .stApp:has(.report-workspace) .report-comp-dot.civilian {
        background: #3d7a8f;
    }

    .stApp:has(.report-workspace) .report-comp-dot.unknown {
        background: #4d5966;
    }

    .stApp:has(.report-workspace) .report-comp-count {
        color: #eef3f8;
        font-size: 0.86rem;
    }

    .stApp:has(.report-workspace) .report-traffic-line {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.82rem;
        color: #9eb8cc;
        letter-spacing: 0.02em;
        margin: 0;
    }

    .stApp:has(.report-workspace) [data-testid="stHorizontalBlock"]:has(.report-figure-detect) [data-testid="stImage"],
    .stApp:has(.report-workspace) [data-testid="stHorizontalBlock"]:has(.report-figure-spatial) [data-testid="stImage"] {
        display: flex;
        justify-content: center;
        width: 100%;
    }

    .stApp:has(.report-workspace) [data-testid="stHorizontalBlock"]:has(.report-figure-detect) [data-testid="stImage"] img {
        display: block;
        width: auto !important;
        max-width: 100% !important;
        height: auto !important;
        max-height: min(46vh, 420px) !important;
        object-fit: contain !important;
        object-position: center center;
    }

    .stApp:has(.report-workspace) [data-testid="stHorizontalBlock"]:has(.report-figure-spatial) [data-testid="stImage"] img {
        display: block;
        width: auto !important;
        max-width: 100% !important;
        height: auto !important;
        max-height: min(38vh, 340px) !important;
        object-fit: contain !important;
        object-position: center center;
        margin: 0 auto;
    }

    .stApp:has(.report-workspace) .report-ai-lead {
        color: #eef3f8;
        font-size: 1.02rem;
        font-weight: 500;
        letter-spacing: -0.1px;
        line-height: 1.35;
        margin: 0.1rem 0 0.85rem 0;
    }

    .stApp:has(.report-workspace) .ai-meta {
        color: #6d8094;
        font-size: 0.72rem;
        margin-bottom: 0.7rem;
    }

    .stApp:has(.report-workspace) .report-risk-banner {
        width: 100%;
        box-sizing: border-box;
        background: #0c1522;
        border: 1px solid #1a2a3d;
        border-radius: 5px;
        padding: 1rem 1.15rem 0.95rem 1.15rem;
        margin: 0.15rem 0 0.2rem 0;
    }

    .stApp:has(.report-workspace) .report-risk-banner.low {
        border-color: #1f4a3a;
        background: #0c1a16;
    }

    .stApp:has(.report-workspace) .report-risk-banner.medium {
        border-color: #6b5420;
        background: #1a160c;
    }

    .stApp:has(.report-workspace) .report-risk-banner.high {
        border-color: #6b3030;
        background: #1a0f0f;
    }

    .stApp:has(.report-workspace) .report-risk-label {
        color: #7d8fa3;
        font-size: 0.6rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.2rem;
    }

    .stApp:has(.report-workspace) .report-risk-level {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.9rem;
        font-weight: 500;
        letter-spacing: 0.08em;
        line-height: 1.1;
        margin: 0 0 0.75rem 0;
    }

    .stApp:has(.report-workspace) .report-risk-banner.low .report-risk-level {
        color: #7dba9a;
    }

    .stApp:has(.report-workspace) .report-risk-banner.medium .report-risk-level {
        color: #d4b56a;
    }

    .stApp:has(.report-workspace) .report-risk-banner.high .report-risk-level {
        color: #d98989;
    }

    .stApp:has(.report-workspace) .report-risk-signals {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.85rem 1.5rem;
        margin-bottom: 0.75rem;
    }

    .stApp:has(.report-workspace) .report-risk-signal {
        min-width: 0;
        padding-left: 0.7rem;
        border-left: 1px solid #1a2a3d;
    }

    .stApp:has(.report-workspace) .report-risk-signal.ok {
        border-left-color: #1f4a3a;
    }

    .stApp:has(.report-workspace) .report-risk-signal.warn {
        border-left-color: #6b5420;
    }

    .stApp:has(.report-workspace) .report-risk-signal.alert {
        border-left-color: #6b3030;
    }

    .stApp:has(.report-workspace) .report-risk-signal-text {
        color: #d8e0ea;
        font-size: 0.92rem;
        line-height: 1.4;
    }

    .stApp:has(.report-workspace) .report-risk-meta {
        color: #6d8094;
        font-size: 0.72rem;
        line-height: 1.35;
        margin: 0;
        padding-top: 0.65rem;
        border-top: 1px solid #152232;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


def reset_analysis():
    st.session_state.analysis_complete = False
    st.session_state.model = None
    st.session_state.results = None
    st.session_state.analysis = None
    st.session_state.orig_img = None
    st.session_state.heatmap = None
    st.session_state.zone_counts = None
    st.session_state.hotspots = None
    st.session_state.zone_overlay = None
    st.session_state.intelligence_report = None
    st.session_state.intelligence_error = None
    st.session_state.pipeline_stage = "mission"


def render_pipeline_indicator(current_stage="mission", completed=False, stages=None):
    stage_list = stages if stages is not None else PIPELINE_STAGES
    stage_ids = [stage_id for stage_id, _ in stage_list]
    current_index = (
        stage_ids.index(current_stage)
        if current_stage in stage_ids
        else 0
    )

    parts = [
        '<div class="pipeline-bar">',
    ]

    for index, (stage_id, label) in enumerate(stage_list):
        if completed:
            state = "done"
        elif index < current_index:
            state = "done"
        elif index == current_index:
            state = "active"
        else:
            state = "pending"

        parts.append('<div class="pipeline-item">')
        parts.append(f'<span class="pipeline-dot {state}"></span>')
        parts.append(
            f'<span class="pipeline-label {state}">{label}</span>'
        )
        parts.append("</div>")

        if index < len(stage_list) - 1:
            parts.append('<div class="pipeline-sep"></div>')

    parts.append("</div>")

    st.markdown("".join(parts), unsafe_allow_html=True)


def to_display_image(image_bgr):
    """Convert OpenCV BGR arrays to RGB for Streamlit display."""

    if image_bgr is None:
        return None

    if len(image_bgr.shape) == 2:
        return image_bgr

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def draw_detection_overlay(results):
    """
    Compact detection visualization for report display.
    Does not alter detection results — presentation only.
    """

    image = results[0].orig_img.copy()
    boxes = results[0].boxes
    names = results[0].names

    if boxes is None or len(boxes) == 0:
        return to_display_image(image)

    height, width = image.shape[:2]
    color = (180, 170, 70)

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        confidence = float(box.conf[0])
        class_name = names[int(box.cls[0])]

        short_name = (
            class_name
            if len(class_name) <= 11
            else class_name[:10] + "."
        )
        label = f"{short_name} {confidence:.2f}"

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color,
            1,
            lineType=cv2.LINE_AA,
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.34
        thickness = 1

        (text_w, text_h), _ = cv2.getTextSize(
            label,
            font,
            font_scale,
            thickness,
        )

        box_height = y2 - y1
        pad = 2

        if box_height > text_h + 8:
            text_x = x1 + pad
            text_y = y1 + text_h + pad + 1
        else:
            text_x = x1
            text_y = y1 - 3

        text_x = int(np.clip(text_x, 0, max(0, width - text_w - 1)))
        text_y = int(np.clip(text_y, text_h + 1, height - 2))

        bg_x1 = max(0, text_x - 1)
        bg_y1 = max(0, text_y - text_h - 2)
        bg_x2 = min(width - 1, text_x + text_w + 1)
        bg_y2 = min(height - 1, text_y + 2)

        cv2.rectangle(
            image,
            (bg_x1, bg_y1),
            (bg_x2, bg_y2),
            (18, 22, 28),
            -1,
        )

        cv2.putText(
            image,
            label,
            (text_x, text_y),
            font,
            font_scale,
            (230, 235, 240),
            thickness,
            lineType=cv2.LINE_AA,
        )

    return to_display_image(image)


def risk_pill_class(level):
    value = str(level).strip().lower()

    if value == "high":
        return "high"

    if value == "medium":
        return "medium"

    return "low"


def render_signal_box(label, text, tone="ok"):
    st.markdown(
        f'<div class="signal-box {tone}">'
        f'<div class="signal-label">{label}</div>'
        f'<div class="signal-text">{text}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def parse_intelligence_report(report):
    sections = {
        "SITUATION ASSESSMENT": "",
        "RISK EXPLANATION": "",
        "AREAS REQUIRING ATTENTION": "",
        "RECOMMENDED ACTION": "",
    }

    if not report:
        return sections

    text = report.strip()

    text = re.sub(
        r"^#+\s*",
        "",
        text,
        flags=re.MULTILINE,
    )

    patterns = [
        "Intelligence Assessment",
        "Risk Explanation",
        "Areas Requiring Attention",
        "Recommended Action",
    ]

    positions = []

    for pattern in patterns:
        match = re.search(
            rf"(?im)^\s*{re.escape(pattern)}\s*$",
            text,
        )

        if match:
            positions.append(
                (
                    match.start(),
                    match.end(),
                    pattern,
                )
            )

    positions.sort()

    if not positions:
        sections["SITUATION ASSESSMENT"] = text
        return sections

    title_map = {
        "Intelligence Assessment": "SITUATION ASSESSMENT",
        "Risk Explanation": "RISK EXPLANATION",
        "Areas Requiring Attention": "AREAS REQUIRING ATTENTION",
        "Recommended Action": "RECOMMENDED ACTION",
    }

    for index, (_, end, title) in enumerate(positions):

        if index + 1 < len(positions):
            content_end = positions[index + 1][0]
        else:
            content_end = len(text)

        content = text[end:content_end].strip()
        content = content.strip(":").strip()

        sections[title_map[title]] = content

    return sections


def render_ai_content(text):
    if not text:
        st.caption("No assessment available")
        return

    for line in text.splitlines():

        value = line.strip()

        if not value:
            continue

        value = re.sub(r"^#+\s*", "", value).strip()

        if not value:
            continue

        if value.lower() in {
            "intelligence assessment",
            "risk explanation",
            "areas requiring attention",
            "recommended action",
            "situation assessment",
        }:
            continue

        if value.startswith("- "):
            st.markdown(value)
            continue

        if re.match(r"^\d+\.\s+", value):
            st.markdown(value)
            continue

        st.write(value)


def build_intelligence_context(analysis, zone_counts, hotspots):
    """
    Merge maritime analysis with spatial intelligence into a single
    structured object for the AI analyst.

    All numpy values are converted to native Python types so the
    data serializes cleanly.
    """

    context = dict(analysis)

    grid_size = int(zone_counts.shape[0])
    total_zones = grid_size * grid_size

    zone_list = zone_counts.tolist()

    max_count = int(zone_counts.max())
    active_zones = int((zone_counts > 0).sum())

    primary_hotspot = None

    if max_count > 0:

        if hasattr(hotspots, "tolist"):
            hotspot_list = hotspots.tolist()

        elif hotspots:
            hotspot_list = [list(item) for item in hotspots]

        else:
            hotspot_list = []

        if hotspot_list:

            primary_hotspot = {
                "row": int(hotspot_list[0][0]) + 1,
                "col": int(hotspot_list[0][1]) + 1,
            }

    context["spatial"] = {
        "grid_size": grid_size,
        "total_zones": total_zones,
        "zone_counts": zone_list,
        "max_zone_concentration": max_count,
        "active_zones": active_zones,
        "primary_hotspot": primary_hotspot,
    }

    return context


if not st.session_state.analysis_complete:

    st.markdown(
        '<div class="opening-workspace"></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="app-kicker">SATELLITE MARITIME INTELLIGENCE</div>',
        unsafe_allow_html=True,
    )

    st.title("Maritime Intelligence")

    pipeline_slot = st.empty()

    with pipeline_slot.container():

        render_pipeline_indicator(
            current_stage="mission",
            completed=False,
            stages=OPENING_PIPELINE_STAGES,
        )

    st.markdown(
        '<div class="opening-intro">'
        '<p class="opening-lead-title">'
        "Turn satellite imagery into a structured maritime intelligence report"
        "</p>"
        '<p class="opening-lead-copy">'
        "Upload a satellite image and the system will detect and classify ships, "
        "analyze vessel composition and traffic, identify spatial concentration, "
        "assess risk signals, and optionally generate an AI-assisted intelligence "
        "assessment"
        "</p>"
        '<div class="opening-steps">'
        '<div class="opening-step">'
        '<div class="opening-step-kicker">01 — DETECT</div>'
        '<p class="opening-step-text">'
        "Ships &amp; boats detected from satellite imagery"
        "</p>"
        "</div>"
        '<div class="opening-step">'
        '<div class="opening-step-kicker">02 — ANALYZE</div>'
        '<p class="opening-step-text">'
        "Vessel composition, traffic, spatial concentration &amp; risk"
        "</p>"
        "</div>"
        '<div class="opening-step">'
        '<div class="opening-step-kicker">03 — INTERPRET</div>'
        '<p class="opening-step-text">'
        "AI-assisted synthesis of the complete intelligence picture"
        "</p>"
        "</div>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload a satellite image to begin",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_file is None:

        st.markdown(
            '<div class="opening-empty"></div>',
            unsafe_allow_html=True,
        )

        st.stop()

    st.markdown(
        '<div class="has-source-image"></div>',
        unsafe_allow_html=True,
    )

    same_stored_file = (
        st.session_state.uploaded_name == uploaded_file.name
        and st.session_state.uploaded_bytes is not None
        and len(st.session_state.uploaded_bytes) == uploaded_file.size
    )

    if same_stored_file:

        uploaded_bytes = st.session_state.uploaded_bytes

    else:

        uploaded_bytes = uploaded_file.getvalue()

        file_signature = hashlib.md5(
            uploaded_bytes
        ).hexdigest()

        if st.session_state.file_signature != file_signature:

            reset_analysis()

            st.session_state.file_signature = file_signature
            st.session_state.uploaded_name = uploaded_file.name
            st.session_state.uploaded_bytes = uploaded_bytes
            st.session_state.pipeline_stage = "mission"

    preview_col, action_col = st.columns(
        [2.4, 1],
        gap="medium",
    )

    with preview_col:

        st.caption("SOURCE IMAGE")

        st.image(
            uploaded_file,
            use_container_width=True,
        )

    with action_col:

        st.caption("ANALYSIS ACTION")

        st.markdown(
            '<p class="opening-ready-note">Source image ready</p>',
            unsafe_allow_html=True,
        )

        analyze_button = st.button(
            "RUN ANALYSIS",
            type="primary",
            use_container_width=True,
        )

    if not analyze_button:
        st.stop()

    def show_pipeline_stage(stage):
        st.session_state.pipeline_stage = stage
        with pipeline_slot.container():
            render_pipeline_indicator(
                current_stage=stage,
                completed=False,
                stages=OPENING_PIPELINE_STAGES,
            )

    try:

        show_pipeline_stage("perception")

        with st.spinner("Running perception..."):

            model = load_model("best.pt")

            with open("temp_image.jpg", "wb") as f:
                f.write(uploaded_bytes)

            results = run_detection(
                model,
                "temp_image.jpg",
            )

        orig_img = results[0].orig_img.copy()

        show_pipeline_stage("analysis")

        with st.spinner("Running maritime analysis..."):

            analysis = generate_analysis(
                results,
                model,
            )

        show_pipeline_stage("spatial")

        with st.spinner("Running spatial intelligence..."):

            heatmap = generate_heatmap(
                results,
                orig_img,
            )

            zone_counts, hotspots, zone_overlay = zone_based_analysis(
                results,
                orig_img,
                grid_size=4,
            )

        show_pipeline_stage("risk")

        st.session_state.model = model
        st.session_state.results = results
        st.session_state.analysis = analysis
        st.session_state.orig_img = orig_img
        st.session_state.heatmap = heatmap
        st.session_state.zone_counts = zone_counts
        st.session_state.hotspots = hotspots
        st.session_state.zone_overlay = zone_overlay
        st.session_state.pipeline_stage = "risk"
        st.session_state.analysis_complete = True

        st.rerun()

    except Exception as error:

        st.session_state.pipeline_stage = "mission"

        st.error(
            "The maritime analysis pipeline could not complete."
        )

        st.caption(
            f"Technical detail: {error}"
        )

        st.stop()


st.markdown(
    '<div class="report-workspace"></div>',
    unsafe_allow_html=True,
)


analysis = st.session_state.analysis
results = st.session_state.results
heatmap = st.session_state.heatmap
zone_counts = st.session_state.zone_counts
hotspots = st.session_state.hotspots
zone_overlay = st.session_state.zone_overlay


header_left, header_right = st.columns(
    [5, 1],
    gap="medium",
)

with header_left:

    st.markdown(
        '<div class="report-kicker">INTELLIGENCE REPORT</div>',
        unsafe_allow_html=True,
    )

with header_right:

    if st.button(
        "NEW ANALYSIS",
        use_container_width=True,
    ):

        reset_analysis()
        st.rerun()


st.markdown(
    '<div class="report-breadcrumb">'
    '<span class="report-crumb">PERCEPTION</span>'
    '<span class="report-crumb-sep">·</span>'
    '<span class="report-crumb">ANALYSIS</span>'
    '<span class="report-crumb-sep">·</span>'
    '<span class="report-crumb">SPATIAL</span>'
    '<span class="report-crumb-sep">·</span>'
    '<span class="report-crumb">RISK</span>'
    '<span class="report-crumb-sep">·</span>'
    '<span class="report-crumb">AI</span>'
    "</div>",
    unsafe_allow_html=True,
)


st.divider()


st.markdown(
    '<div class="stage-kicker">01  /  PERCEPTION</div>',
    unsafe_allow_html=True,
)

st.header("Vessel Observation")


detection_col1, detection_col2 = st.columns(
    [1.45, 1],
    gap="medium",
)


with detection_col1:

    st.markdown(
        '<div class="report-figure-detect"></div>',
        unsafe_allow_html=True,
    )

    st.caption("ANNOTATED DETECTION")

    detection_image = draw_detection_overlay(results)

    st.image(
        detection_image,
        use_container_width=True,
    )


with detection_col2:

    st.markdown(
        f'<div class="report-primary-stat">'
        f'<div class="report-primary-value">{analysis["total_ships"]}</div>'
        f'<div class="report-primary-label">VESSELS DETECTED</div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    if len(results[0].boxes) > 0:

        confidences = results[0].boxes.conf.cpu().numpy()

        average_confidence = float(confidences.mean()) * 100
        min_confidence = float(confidences.min()) * 100
        max_confidence = float(confidences.max()) * 100

        st.markdown(
            '<div class="report-supporting">'
            '<div class="config-line">'
            '<span class="config-key">Confidence average</span>'
            f'<span class="config-val">{average_confidence:.1f}%</span>'
            "</div>"
            '<div class="config-line">'
            '<span class="config-key">Confidence range</span>'
            f'<span class="config-val">'
            f"{min_confidence:.1f}% – {max_confidence:.1f}%"
            "</span>"
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    else:

        st.caption("No vessels detected in this image")

    if analysis["class_breakdown"]:

        class_rows = []

        for vessel_type, count in analysis["class_breakdown"].items():

            class_rows.append(
                '<div class="class-row">'
                f'<span class="class-name">{vessel_type}</span>'
                f'<span class="class-count">{count}</span>'
                "</div>"
            )

        st.markdown(
            '<div class="report-supporting">'
            + "".join(class_rows)
            + "</div>",
            unsafe_allow_html=True,
        )

    else:

        st.write("No classifications available")


st.divider()


st.markdown(
    '<div class="stage-kicker">02  /  MARITIME ANALYSIS</div>',
    unsafe_allow_html=True,
)

st.header("Composition & Traffic")

military_count = analysis["military_ships"]
civilian_count = analysis["civilian_ships"]
unknown_count = analysis["unknown_ships"]
composition_total = military_count + civilian_count + unknown_count

if composition_total > 0:
    military_flex = military_count
    civilian_flex = civilian_count
    unknown_flex = unknown_count
else:
    military_flex = 0
    civilian_flex = 0
    unknown_flex = 0

military_seg = (
    f'<div class="report-comp-seg military" style="flex:{military_flex};min-width:6px"></div>'
    if military_count > 0
    else ""
)
civilian_seg = (
    f'<div class="report-comp-seg civilian" style="flex:{civilian_flex};min-width:6px"></div>'
    if civilian_count > 0
    else ""
)
unknown_seg = (
    f'<div class="report-comp-seg unknown" style="flex:{unknown_flex};min-width:6px"></div>'
    if unknown_count > 0
    else ""
)

st.markdown(
    f'<div class="report-comp">'
    f'<div class="report-comp-heading">COMPOSITION</div>'
    f'<div class="report-comp-bar">{military_seg}{civilian_seg}{unknown_seg}</div>'
    f'<div class="report-comp-legend">'
    f'<div class="report-comp-item">'
    f'<span class="report-comp-dot military"></span>'
    f"MILITARY "
    f'<span class="report-comp-count">{military_count}</span>'
    f"</div>"
    f'<div class="report-comp-item">'
    f'<span class="report-comp-dot civilian"></span>'
    f"CIVILIAN "
    f'<span class="report-comp-count">{civilian_count}</span>'
    f"</div>"
    f'<div class="report-comp-item">'
    f'<span class="report-comp-dot unknown"></span>'
    f"UNKNOWN "
    f'<span class="report-comp-count">{unknown_count}</span>'
    f"</div>"
    f"</div>"
    f'<div class="report-comp-heading">TRAFFIC</div>'
    f'<p class="report-traffic-line">'
    f'{analysis["congestion"]["level"]} congestion · '
    f'{analysis["congestion"]["density"]} density'
    f"</p>"
    f"</div>",
    unsafe_allow_html=True,
)


st.divider()


st.markdown(
    '<div class="stage-kicker">03  /  SPATIAL INTELLIGENCE</div>',
    unsafe_allow_html=True,
)

st.header("Activity Concentration")


spatial_col1, spatial_col2 = st.columns(
    2,
    gap="medium",
)


with spatial_col1:

    st.markdown(
        '<div class="report-figure-spatial"></div>',
        unsafe_allow_html=True,
    )

    st.caption("VESSEL DENSITY")

    st.image(
        to_display_image(heatmap),
        use_container_width=True,
    )


with spatial_col2:

    st.markdown(
        '<div class="report-figure-spatial"></div>',
        unsafe_allow_html=True,
    )

    st.caption("TRAFFIC ZONES")

    st.image(
        to_display_image(zone_overlay),
        use_container_width=True,
    )


if zone_counts.max() > 0:

    active_zones = int(
        (zone_counts > 0).sum()
    )

    highest_concentration = int(
        zone_counts.max()
    )

    primary_hotspot = hotspots[0]

    hotspot_row = int(
        primary_hotspot[0]
    ) + 1

    hotspot_col = int(
        primary_hotspot[1]
    ) + 1

    st.markdown(
        f'<div class="report-spatial-meta">'
        f'<div class="report-spatial-item">'
        f'<div class="report-spatial-label">PRIMARY HOTSPOT</div>'
        f'<div class="report-spatial-value">'
        f"Zone {hotspot_row},{hotspot_col}"
        f"</div>"
        f"</div>"
        f'<div class="report-spatial-item">'
        f'<div class="report-spatial-label">MAX CONCENTRATION</div>'
        f'<div class="report-spatial-value">{highest_concentration}</div>'
        f"</div>"
        f'<div class="report-spatial-item">'
        f'<div class="report-spatial-label">ACTIVE ZONES</div>'
        f'<div class="report-spatial-value">{active_zones} / 16</div>'
        f"</div>"
        f"</div>"
        f'<p class="report-spatial-note">'
        f"Highest concentration in Zone {hotspot_row},{hotspot_col} "
        f"({highest_concentration} vessels). "
        f"Activity present in {active_zones} of 16 zones."
        f"</p>",
        unsafe_allow_html=True,
    )


st.divider()


st.markdown(
    '<div class="stage-kicker">04  /  RISK ASSESSMENT</div>',
    unsafe_allow_html=True,
)

st.header("Attention Signals")


risk_level = analysis["risk_level"]
risk_class = risk_pill_class(risk_level)

if analysis["clustering"]["detected"]:
    clustering_status = "DETECTED"
    clustering_tone = "warn"
else:
    clustering_status = "NOT DETECTED"
    clustering_tone = "ok"

alert_text = analysis["alert"]

if "CRITICAL" in alert_text.upper():
    alert_tone = "alert"
elif "WARNING" in alert_text.upper():
    alert_tone = "warn"
else:
    alert_tone = "ok"


alert_text_html = (
    str(alert_text)
    .rstrip(".")
    .replace("&", "&amp;")
    .replace("<", "&lt;")
    .replace(">", "&gt;")
)
clustering_status_html = (
    str(clustering_status)
    .replace("&", "&amp;")
    .replace("<", "&lt;")
    .replace(">", "&gt;")
)
clustering_message_html = (
    str(analysis["clustering"]["message"])
    .rstrip(".")
    .replace("&", "&amp;")
    .replace("<", "&lt;")
    .replace(">", "&gt;")
)
risk_level_html = (
    str(risk_level)
    .upper()
    .replace("&", "&amp;")
    .replace("<", "&lt;")
    .replace(">", "&gt;")
)

st.markdown(
    f'<div class="report-risk-banner {risk_class}">'
    f'<div class="report-risk-level">{risk_level_html} RISK</div>'
    f'<div class="report-risk-signals">'
    f'<div class="report-risk-signal {alert_tone}">'
    f'<div class="report-risk-signal-text">{alert_text_html}</div>'
    f"</div>"
    f'<div class="report-risk-signal {clustering_tone}">'
    f'<div class="report-risk-signal-text">{clustering_status_html}</div>'
    f"</div>"
    f"</div>"
    f'<p class="report-risk-meta">'
    f"Military vessels: {analysis['military_ships']} · "
    f"Total vessels: {analysis['total_ships']} · "
    f"{clustering_message_html}"
    f"</p>"
    f"</div>",
    unsafe_allow_html=True,
)


st.divider()


st.markdown(
    '<div class="stage-kicker">05  /  AI ANALYST</div>',
    unsafe_allow_html=True,
)

st.header("AI Intelligence Assessment")

st.markdown(
    '<p class="report-ai-lead">'
    "Generate an interpretation of the complete evidence above"
    "</p>",
    unsafe_allow_html=True,
)


if st.session_state.intelligence_report is None:

    with st.container(border=True):

        if st.session_state.intelligence_error:

            error_message = st.session_state.intelligence_error

            if "quota" in error_message.lower():

                st.warning(
                    "AI assessment unavailable: the Gemini API quota "
                    "for this key has been reached. No further calls "
                    "were attempted. Retry after the quota resets."
                )

            else:

                st.warning(
                    "AI intelligence assessment is temporarily unavailable. "
                    "The computer vision analysis remains available."
                )

            st.caption(
                f"Technical detail: {error_message}"
            )

            generate_button = st.button(
                "RETRY AI ASSESSMENT",
                type="primary",
                use_container_width=True,
            )

        else:

            generate_button = st.button(
                "GENERATE AI ASSESSMENT",
                type="primary",
                use_container_width=True,
            )

        if generate_button:

            try:

                with st.spinner(
                    "Generating intelligence assessment..."
                ):

                    intelligence_context = build_intelligence_context(
                        analysis,
                        zone_counts,
                        hotspots,
                    )

                    st.session_state.intelligence_report = (
                        generate_intelligence_report(
                            intelligence_context
                        )
                    )

                    st.session_state.intelligence_error = None

            except GeminiQuotaError as error:

                st.session_state.intelligence_error = str(error)

                st.rerun()

            except Exception as error:

                st.session_state.intelligence_error = str(error)

                st.rerun()


if st.session_state.intelligence_report:

    report_sections = parse_intelligence_report(
        st.session_state.intelligence_report
    )

    ai_col1, ai_col2 = st.columns(
        2,
        gap="medium",
    )

    with ai_col1:

        with st.container(border=True):

            st.markdown(
                '<div class="ai-panel-label">SITUATION ASSESSMENT</div>',
                unsafe_allow_html=True,
            )

            render_ai_content(
                report_sections[
                    "SITUATION ASSESSMENT"
                ]
            )

        with st.container(border=True):

            st.markdown(
                '<div class="ai-panel-label">RISK EXPLANATION</div>',
                unsafe_allow_html=True,
            )

            render_ai_content(
                report_sections[
                    "RISK EXPLANATION"
                ]
            )

    with ai_col2:

        with st.container(border=True):

            st.markdown(
                '<div class="ai-panel-label">AREAS REQUIRING ATTENTION</div>',
                unsafe_allow_html=True,
            )

            render_ai_content(
                report_sections[
                    "AREAS REQUIRING ATTENTION"
                ]
            )

        with st.container(border=True):

            st.markdown(
                '<div class="ai-panel-label">RECOMMENDED ACTION</div>',
                unsafe_allow_html=True,
            )

            render_ai_content(
                report_sections[
                    "RECOMMENDED ACTION"
                ]
            )

st.divider()


st.caption("MISSION SUMMARY")

st.markdown(
    f'<div class="mono-value">'
    f"{analysis['total_ships']} vessels · "
    f"{analysis['military_ships']} military · "
    f"{analysis['civilian_ships']} civilian · "
    f"{analysis['congestion']['level']} congestion · "
    f"{analysis['risk_level']} risk"
    f"</div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="obs-note">'
    "Computer vision observations should be verified by a human analyst "
    "before operational decisions are made."
    "</p>",
    unsafe_allow_html=True,
)
