"""HTML presentation fragments for the Streamlit dashboard (no business logic)."""

_ICON_DETECT = """
<svg class="opening-step-svg" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <circle cx="11" cy="11" r="7" stroke="currentColor" stroke-width="1.5"/>
  <path d="M11 4v2M11 16v2M4 11h2M16 11h2" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
  <path d="M20 20l-3-3" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
</svg>
"""

_ICON_ANALYZE = """
<svg class="opening-step-svg" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <rect x="3" y="12" width="4" height="8" rx="1" stroke="currentColor" stroke-width="1.5"/>
  <rect x="10" y="8" width="4" height="12" rx="1" stroke="currentColor" stroke-width="1.5"/>
  <rect x="17" y="5" width="4" height="15" rx="1" stroke="currentColor" stroke-width="1.5"/>
</svg>
"""

_ICON_INTERPRET = """
<svg class="opening-step-svg" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <path d="M12 3l1.8 4.2L18 9l-4.2 1.8L12 15l-1.8-4.2L6 9l4.2-1.8L12 3z" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"/>
  <path d="M5 19c2-2 4.5-3 7-3s5 1 7 3" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
</svg>
"""

_BRAND_MARK = """
<svg class="opening-brand-svg" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <rect x="2" y="2" width="28" height="28" rx="8" stroke="currentColor" stroke-width="1.5"/>
  <path d="M8 20c2-4 5-6 8-6s6 2 8 6" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
  <circle cx="16" cy="12" r="2" fill="currentColor"/>
</svg>
"""


def opening_header_html() -> str:
    return (
        '<div class="opening-page-head">'
        '<header class="opening-topbar">'
        '<p class="opening-kicker">Satellite maritime intelligence</p>'
        '<div class="opening-topbar-row">'
        '<div class="opening-brand">'
        f'<span class="opening-brand-mark">{_BRAND_MARK}</span>'
        '<h1 class="opening-hero-title">Maritime Intelligence</h1>'
        "</div>"
        '<div class="opening-status-chip">'
        '<span class="opening-status-dot"></span>'
        "<span>Mission · Standby</span>"
        "</div>"
        "</div>"
        "</header>"
        "</div>"
    )


def opening_intro_html() -> str:
    return (
        '<div class="opening-intro">'
        '<p class="opening-lead-title">'
        "Turn satellite imagery into a structured maritime intelligence report"
        "</p>"
        '<p class="opening-lead-copy">'
        "Upload a satellite image and the system will detect and classify ships, "
        "analyze vessel composition and traffic, identify spatial concentration, "
        "assess risk signals, and optionally generate an AI-assisted intelligence "
        "assessment."
        "</p>"
        '<div class="opening-steps">'
        '<article class="opening-step opening-step-card">'
        f'<div class="opening-step-icon">{_ICON_DETECT}</div>'
        '<div class="opening-step-body">'
        '<div class="opening-step-kicker">01 · Detect</div>'
        '<p class="opening-step-text">'
        "Ships and boats located in satellite imagery"
        "</p>"
        "</div>"
        "</article>"
        '<article class="opening-step opening-step-card">'
        f'<div class="opening-step-icon">{_ICON_ANALYZE}</div>'
        '<div class="opening-step-body">'
        '<div class="opening-step-kicker">02 · Analyze</div>'
        '<p class="opening-step-text">'
        "Composition, traffic, spatial concentration, and risk signals"
        "</p>"
        "</div>"
        "</article>"
        '<article class="opening-step opening-step-card">'
        f'<div class="opening-step-icon">{_ICON_INTERPRET}</div>'
        '<div class="opening-step-body">'
        '<div class="opening-step-kicker">03 · Interpret</div>'
        '<p class="opening-step-text">'
        "AI-assisted synthesis of the full intelligence picture"
        "</p>"
        "</div>"
        "</article>"
        "</div>"
        "</div>"
    )


def opening_upload_prompt_html() -> str:
    return (
        '<div class="opening-upload-prompt">'
        '<p class="opening-upload-title">Upload satellite imagery</p>'
        '<p class="opening-upload-hint">'
        "Drag and drop or use the control below · JPG or PNG"
        "</p>"
        "</div>"
    )


def opening_preview_shell_html() -> str:
    return '<div class="opening-preview-card"></div>'


def opening_action_shell_html() -> str:
    return (
        '<div class="opening-action-card">'
        '<p class="opening-action-eyebrow">Analysis</p>'
        '<p class="opening-ready-note">Source image loaded and ready to process.</p>'
        "</div>"
    )


def pipeline_indicator_html(
    current_stage="mission",
    completed=False,
    stages=None,
) -> str:
    if not stages:
        return ""

    stage_ids = [stage_id for stage_id, _ in stages]
    current_index = stage_ids.index(current_stage) if current_stage in stage_ids else 0

    parts = ['<div class="pipeline-bar">']

    for index, (stage_id, label) in enumerate(stages):
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
        parts.append(f'<span class="pipeline-label {state}">{label}</span>')
        parts.append("</div>")

        if index < len(stages) - 1:
            parts.append('<div class="pipeline-sep"></div>')

    parts.append("</div>")
    return "".join(parts)


def report_stage_rail_html(stages) -> str:
    """Single-line report stage labels (no second pipeline row)."""
    parts = ['<div class="report-stage-rail">']
    for index, (_, label) in enumerate(stages):
        parts.append(f'<span class="report-stage-chip">{label}</span>')
        if index < len(stages) - 1:
            parts.append('<span class="report-stage-sep">·</span>')
    parts.append("</div>")
    return "".join(parts)


def report_sticky_nav_html(stages) -> str:
    report_stages = [
        (stage_id, label)
        for stage_id, label in stages
        if stage_id != "mission"
    ]
    rail = report_stage_rail_html(report_stages)
    return (
        '<div class="report-sticky-nav">'
        f'<div class="report-sticky-inner">{rail}</div>'
        "</div>"
    )


def report_section_header_html(
    number: str,
    code: str,
    title: str,
    anchor_class: str,
) -> str:
    return (
        f'<div class="report-section-anchor {anchor_class}"></div>'
        '<div class="report-section-header">'
        f'<div class="stage-kicker">{number}  /  {code}</div>'
        f'<h2 class="report-section-title">{title}</h2>'
        "</div>"
    )


def report_footer_html(
    total_ships: int,
    military_ships: int,
    civilian_ships: int,
    congestion_level: str,
    risk_level: str,
) -> str:
    return (
        '<div class="report-footer">'
        '<div class="report-footer-label">Mission summary</div>'
        '<div class="mono-value report-footer-stats">'
        f"{total_ships} vessels · "
        f"{military_ships} military · "
        f"{civilian_ships} civilian · "
        f"{congestion_level} congestion · "
        f"{risk_level} risk"
        "</div>"
        '<p class="obs-note report-footer-note">'
        "Computer vision observations should be verified by a human analyst "
        "before operational decisions are made."
        "</p>"
        "</div>"
    )
