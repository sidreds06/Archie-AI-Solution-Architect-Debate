"""Gradio Blocks layout for Archie smart UI.

Defines all components across all phases: welcome, interview, debate, verdict.
"""

from __future__ import annotations

import gradio as gr

from config import ADVERSARY_MODEL, MODERATOR_MODEL, PROPOSER_MODEL
from ui.themes import build_theme


def build_layout() -> tuple[gr.Blocks, object, str, str]:
    """Construct and return (app, theme, css, js) for Gradio 6+ launch()."""
    theme = build_theme()
    custom_css = _build_css()
    custom_js = _build_js()

    with gr.Blocks(
        title="Archie — AI Solution Architect Debate",
    ) as app:
        # ---- Shared state ----
        engine_state = gr.State(None)  # Will hold the DebateEngine instance

        # ========================================
        # PHASE 1: Welcome
        # ========================================
        with gr.Column(visible=True, elem_id="welcome-phase") as welcome_phase:
            gr.Markdown(
                "# 🏛️ Archie — AI Solution Architect Debate\n"
                "Two AI agents debate your architecture problem. A third moderates and scores.",
                elem_id="main-title",
            )
            gr.Markdown(
                f"**Proposer:** `{PROPOSER_MODEL}` &nbsp;|&nbsp; "
                f"**Adversary:** `{ADVERSARY_MODEL}` &nbsp;|&nbsp; "
                f"**Moderator:** `{MODERATOR_MODEL}`",
                elem_id="model-roster",
            )
            memory_display = gr.HTML(value="", elem_id="memory-display")

            problem_input = gr.Textbox(
                label="Describe your architecture problem",
                placeholder="e.g., Design a real-time data pipeline for 500k events/sec with sub-100ms latency...",
                lines=4,
                max_lines=8,
            )
            with gr.Row():
                rounds_input = gr.Slider(
                    minimum=1, maximum=10, value=5, step=1,
                    label="Debate Rounds",
                )
                start_btn = gr.Button(
                    "🚀 Start Debate",
                    variant="primary",
                    size="lg",
                )

        # ========================================
        # PHASE 2: Interview
        # ========================================
        with gr.Column(visible=False, elem_id="interview-phase") as interview_phase:
            gr.Markdown("## 📋 Requirements Interview")
            interview_chat = gr.Chatbot(
                label="Interview",
                height=300,
                elem_id="interview-chat",
            )
            interview_input = gr.Textbox(
                label="Your answer",
                placeholder="Type your answer...",
                lines=2,
            )
            with gr.Row():
                interview_submit = gr.Button("Submit Answer", variant="primary")
                interview_skip = gr.Button("Skip Question", variant="secondary")
            interview_progress = gr.Markdown("", elem_id="interview-progress")

        # ========================================
        # PHASE 3: Active Debate
        # ========================================
        with gr.Column(visible=False, elem_id="debate-phase") as debate_phase:
            # Header bar
            debate_header = gr.Markdown("## ⚔️ DEBATE", elem_id="debate-header")

            with gr.Row():
                # ---- LEFT SIDEBAR (30%) ----
                with gr.Column(scale=3, min_width=280, elem_id="sidebar"):
                    gr.HTML('<div class="sidebar-label">Scoreboard</div>', elem_id="sidebar-title")
                    scoreboard_html = gr.HTML(value="", elem_id="scoreboard")

                    gr.HTML('<div class="sidebar-label">Rubric</div>', elem_id="rubric-title")
                    rubric_html = gr.HTML(value="", elem_id="rubric")

                    gr.HTML('<div class="sidebar-label">Momentum</div>', elem_id="momentum-title")
                    momentum_html = gr.HTML(value="", elem_id="momentum")

                    gr.HTML('<div class="sidebar-label">Events</div>', elem_id="events-title")
                    events_html = gr.HTML(value="", elem_id="events")

                    gr.HTML('<div class="sidebar-label">Search</div>', elem_id="search-title")
                    search_html = gr.HTML(value="", elem_id="search-tracker")

                    score_trend_chart = gr.Plot(
                        label="Score Trend",
                        visible=False,
                        elem_id="score-trend",
                    )

                # ---- RIGHT MAIN AREA (70%) ----
                with gr.Column(scale=7, elem_id="debate-main"):
                    # Dynamic accordion panels for each agent turn
                    # These are created dynamically per round, but we need
                    # pre-built slots for the current round's turns.

                    # Status bar at top
                    debate_status = gr.HTML(value="", elem_id="debate-status")

                    # Agent panels — we use a Markdown area that accumulates
                    # collapsible sections via HTML <details> tags
                    debate_feed = gr.HTML(
                        value="",
                        elem_id="debate-feed",
                    )

                    # Current streaming panel (active agent)
                    with gr.Accordion(
                        label="🔵 Waiting...", open=True, visible=False,
                        elem_id="streaming-accordion",
                    ) as streaming_accordion:
                        streaming_content = gr.Markdown(
                            value="",
                            elem_id="streaming-content",
                        )

                    # Moderator compact card
                    moderator_card = gr.HTML(
                        value="", visible=False,
                        elem_id="moderator-card",
                    )

                    # Dramatic event banner
                    event_banner = gr.HTML(value="", visible=False, elem_id="event-banner")

                    # HITL section
                    with gr.Column(visible=False, elem_id="hitl-section") as hitl_section:
                        gr.Markdown("### ⚠️ MODERATOR NEEDS YOUR INPUT")
                        hitl_context = gr.Markdown("", elem_id="hitl-context")
                        hitl_input = gr.Textbox(
                            label="Your answer",
                            placeholder="Type your answer...",
                            lines=2,
                        )
                        with gr.Row():
                            hitl_submit = gr.Button("Submit Answer", variant="primary")
                            hitl_skip = gr.Button("Skip", variant="secondary")

                    # Round checkpoint
                    with gr.Column(visible=False, elem_id="checkpoint-section") as checkpoint_section:
                        checkpoint_info = gr.Markdown("", elem_id="checkpoint-info")
                        direction_input = gr.Textbox(
                            label="Add direction for the agents (optional)",
                            placeholder="Press Enter to continue, or type direction...",
                            lines=1,
                        )
                        continue_btn = gr.Button("Continue ▶", variant="primary")

        # ========================================
        # PHASE 4: Verdict
        # ========================================
        with gr.Column(visible=False, elem_id="verdict-phase") as verdict_phase:
            gr.Markdown("## 🏆 VERDICT")

            with gr.Row():
                # Left column — stats
                with gr.Column(scale=3, min_width=280):
                    verdict_scores = gr.Markdown("", elem_id="verdict-scores")
                    verdict_radar = gr.Plot(label="Rubric Comparison", elem_id="verdict-radar")
                    verdict_trend = gr.Plot(label="Score History", elem_id="verdict-trend")
                    verdict_stats = gr.Markdown("", elem_id="verdict-stats")
                    verdict_memory_diff = gr.HTML(value="", elem_id="verdict-memory-diff")

                # Right column — brief
                with gr.Column(scale=7):
                    verdict_brief = gr.Markdown("", elem_id="verdict-brief")
                    with gr.Row():
                        export_md_btn = gr.Button("📄 Export Markdown")
                        export_json_btn = gr.Button("💾 Export JSON")
                    export_file = gr.File(label="Download", visible=False, elem_id="export-file")

        # ---- Bottom status bar ----
        status_bar = gr.HTML(
            value='<div style="color: #64748b; padding: 4px 12px; font-size: 12px;">Ready</div>',
            elem_id="status-bar",
        )

        # ========================================
        # Store component references for handlers
        # ========================================
        app.components_map = {
            "engine_state": engine_state,
            # Welcome
            "welcome_phase": welcome_phase,
            "memory_display": memory_display,
            "problem_input": problem_input,
            "rounds_input": rounds_input,
            "start_btn": start_btn,
            # Interview
            "interview_phase": interview_phase,
            "interview_chat": interview_chat,
            "interview_input": interview_input,
            "interview_submit": interview_submit,
            "interview_skip": interview_skip,
            "interview_progress": interview_progress,
            # Debate
            "debate_phase": debate_phase,
            "debate_header": debate_header,
            "scoreboard_html": scoreboard_html,
            "rubric_html": rubric_html,
            "momentum_html": momentum_html,
            "events_html": events_html,
            "search_html": search_html,
            "score_trend_chart": score_trend_chart,
            "debate_status": debate_status,
            "debate_feed": debate_feed,
            "streaming_accordion": streaming_accordion,
            "streaming_content": streaming_content,
            "moderator_card": moderator_card,
            "event_banner": event_banner,
            "hitl_section": hitl_section,
            "hitl_context": hitl_context,
            "hitl_input": hitl_input,
            "hitl_submit": hitl_submit,
            "hitl_skip": hitl_skip,
            "checkpoint_section": checkpoint_section,
            "checkpoint_info": checkpoint_info,
            "direction_input": direction_input,
            "continue_btn": continue_btn,
            # Verdict
            "verdict_phase": verdict_phase,
            "verdict_scores": verdict_scores,
            "verdict_radar": verdict_radar,
            "verdict_trend": verdict_trend,
            "verdict_stats": verdict_stats,
            "verdict_memory_diff": verdict_memory_diff,
            "verdict_brief": verdict_brief,
            "export_md_btn": export_md_btn,
            "export_json_btn": export_json_btn,
            "export_file": export_file,
            # Status
            "status_bar": status_bar,
        }

        # Wire events inside the Blocks context
        from ui.handlers import wire_events
        wire_events(app)

    return app, theme, custom_css, custom_js


def _build_css() -> str:
    """Custom CSS for the Archie UI."""
    return """
    /* ===== GLOBAL ===== */
    .gradio-container { padding: 8px 12px !important; }

    /* ===== WELCOME PHASE ===== */
    #welcome-phase { max-width: 700px; margin: 48px auto; padding: 0 16px; }
    #main-title { text-align: center; margin-bottom: 6px; }
    #main-title h1 { font-size: 26px !important; font-weight: 700 !important; letter-spacing: -0.5px; }
    #main-title p  { color: #94a3b8 !important; font-size: 14px !important; }
    #model-roster { text-align: center; }
    #model-roster p { color: #475569 !important; font-size: 12px !important; margin: 0 !important; }

    /* ===== INTERVIEW PHASE ===== */
    #interview-phase { max-width: 800px; margin: 0 auto; padding: 16px; }
    #interview-progress p { color: #64748b !important; font-size: 12px !important; margin: 2px 0 !important; }

    /* ===== DEBATE HEADER ===== */
    #debate-header { padding-bottom: 6px; }
    #debate-header h2 { font-size: 16px !important; margin: 0 !important; padding-bottom: 0 !important; }

    /* ===== SIDEBAR ===== */
    #sidebar {
        border-right: 1px solid rgba(255,255,255,0.07);
        padding-right: 12px;
        max-height: calc(100vh - 120px);
        overflow-y: auto;
        scrollbar-width: thin;
        scrollbar-color: rgba(100,100,100,0.25) transparent;
    }
    #sidebar::-webkit-scrollbar { width: 4px; }
    #sidebar::-webkit-scrollbar-thumb { background: rgba(100,100,100,0.25); border-radius: 2px; }

    .sidebar-label {
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #475569;
        padding: 8px 0 3px;
    }

    /* Strip Gradio default padding from sidebar wrappers */
    #sidebar-title, #rubric-title, #momentum-title, #events-title, #search-title,
    #scoreboard, #rubric, #momentum, #events, #search-tracker {
        padding: 0 !important;
        margin: 0 !important;
    }

    /* ===== DEBATE FEED ===== */
    #debate-feed {
        max-height: 560px;
        overflow-y: auto;
        padding: 2px 0;
        scrollbar-width: thin;
        scrollbar-color: rgba(100,100,100,0.25) transparent;
    }
    #debate-feed::-webkit-scrollbar { width: 6px; }
    #debate-feed::-webkit-scrollbar-track { background: transparent; }
    #debate-feed::-webkit-scrollbar-thumb { background: rgba(100,100,100,0.3); border-radius: 3px; }

    /* ===== AGENT PANELS ===== */
    .agent-panel {
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 6px;
        margin: 3px 0;
        overflow: hidden;
        background: rgba(0,0,0,0.12);
    }
    .agent-panel:hover { border-color: rgba(255,255,255,0.13); }
    .agent-panel summary {
        padding: 7px 12px;
        cursor: pointer;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.02em;
        display: flex;
        align-items: center;
        gap: 6px;
        list-style: none;
        user-select: none;
    }
    .agent-panel summary::-webkit-details-marker { display: none; }
    .agent-panel summary::before {
        content: "▶";
        font-size: 8px;
        opacity: 0.4;
        transition: transform 0.15s;
        flex-shrink: 0;
    }
    .agent-panel[open] > summary::before { transform: rotate(90deg); }
    .agent-panel summary:hover { background: rgba(255,255,255,0.04); }
    .agent-panel .panel-content {
        padding: 10px 14px 12px;
        border-top: 1px solid rgba(255,255,255,0.06);
        font-size: 13px;
        line-height: 1.65;
    }

    /* Agent color left borders */
    .agent-panel.proposer  { border-left: 3px solid #3b82f6; }
    .agent-panel.adversary { border-left: 3px solid #22c55e; }
    .agent-panel.moderator { border-left: 3px solid #a855f7; }
    .agent-panel.deep-dive { border-left: 3px solid #06b6d4; }

    /* ===== MARKDOWN INSIDE AGENT PANELS ===== */
    .panel-content h1, .panel-content h2, .panel-content h3,
    .panel-content h4, .panel-content h5, .panel-content h6 {
        margin: 10px 0 5px; font-weight: 600; color: #e2e8f0; line-height: 1.3;
    }
    .panel-content h1 { font-size: 16px; }
    .panel-content h2 { font-size: 15px; border-bottom: 1px solid rgba(255,255,255,0.07); padding-bottom: 4px; }
    .panel-content h3 { font-size: 14px; }
    .panel-content h4, .panel-content h5, .panel-content h6 { font-size: 13px; }
    .panel-content p   { margin: 5px 0; color: #cbd5e1; }
    .panel-content ul, .panel-content ol { margin: 5px 0 5px 18px; color: #cbd5e1; }
    .panel-content li  { margin: 2px 0; }
    .panel-content li > ul, .panel-content li > ol { margin: 2px 0 2px 14px; }
    .panel-content strong { color: #f1f5f9; font-weight: 600; }
    .panel-content em     { color: #94a3b8; }
    .panel-content a      { color: #60a5fa; }
    .panel-content blockquote {
        border-left: 3px solid rgba(255,255,255,0.15);
        padding: 4px 0 4px 12px;
        margin: 8px 0;
        color: #94a3b8;
    }
    .panel-content code {
        background: rgba(255,255,255,0.08);
        padding: 1px 5px;
        border-radius: 3px;
        font-size: 12px;
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        color: #e2e8f0;
    }
    .panel-content pre {
        background: rgba(0,0,0,0.35);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 5px;
        padding: 10px 12px;
        margin: 8px 0;
        overflow-x: auto;
        scrollbar-width: thin;
    }
    .panel-content pre code { background: none; padding: 0; font-size: 12px; }
    .panel-content table { width: 100%; border-collapse: collapse; font-size: 12px; margin: 8px 0; }
    .panel-content th {
        background: rgba(255,255,255,0.06);
        padding: 5px 8px;
        text-align: left;
        color: #e2e8f0;
        font-weight: 600;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .panel-content td { padding: 5px 8px; color: #cbd5e1; border-top: 1px solid rgba(255,255,255,0.05); }
    .panel-content hr { border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 10px 0; }

    /* ===== STREAMING ACCORDION ===== */
    #streaming-accordion {
        border-radius: 6px !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        margin: 4px 0 !important;
    }
    #streaming-content {
        max-height: 320px;
        overflow-y: auto;
        font-size: 13px;
        line-height: 1.65;
        color: #cbd5e1;
        scrollbar-width: thin;
    }

    /* ===== MODERATOR CARD ===== */
    .moderator-card {
        background: rgba(168, 85, 247, 0.05);
        border: 1px solid rgba(168, 85, 247, 0.2);
        border-left: 3px solid #a855f7;
        border-radius: 6px;
        padding: 9px 13px;
        margin: 4px 0;
        font-size: 12px;
    }
    #moderator-card { padding: 0 !important; }

    /* ===== HITL SECTION ===== */
    #hitl-section {
        background: rgba(234, 179, 8, 0.04);
        border: 1px solid rgba(234, 179, 8, 0.22);
        border-left: 3px solid #eab308;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 6px 0;
    }
    #hitl-section h3 {
        font-size: 12px !important;
        font-weight: 700 !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
        color: #fbbf24 !important;
        margin: 0 0 8px !important;
    }
    #hitl-context p { font-size: 13px !important; }

    /* ===== CHECKPOINT SECTION ===== */
    #checkpoint-section {
        background: rgba(59, 130, 246, 0.04);
        border: 1px solid rgba(59, 130, 246, 0.18);
        border-left: 3px solid #3b82f6;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 6px 0;
    }
    #checkpoint-info p      { font-size: 13px !important; color: #94a3b8 !important; margin: 2px 0 !important; }
    #checkpoint-info strong { color: #93c5fd !important; }

    /* ===== STATUS BAR ===== */
    #status-bar { border-top: 1px solid rgba(255,255,255,0.06); margin-top: 6px; }

    /* ===== VERDICT PHASE ===== */
    #verdict-brief {
        max-height: 580px;
        overflow-y: auto;
        padding: 16px 18px;
        background: rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 8px;
        font-size: 14px;
        line-height: 1.7;
        scrollbar-width: thin;
    }
    #verdict-scores p { margin: 4px 0 !important; }
    #verdict-stats p  { margin: 3px 0 !important; font-size: 13px !important; }

    /* ===== MEMORY DISPLAY ===== */
    #memory-display { margin-bottom: 4px; }
    """


def _build_js() -> str:
    """JavaScript run on page load — auto-scrolls the debate feed as new content arrives."""
    return """
function() {
    const waitForFeed = () => {
        const feed = document.querySelector('#debate-feed');
        if (feed) {
            new MutationObserver(() => { feed.scrollTop = feed.scrollHeight; })
                .observe(feed, { childList: true, subtree: true });
        } else {
            setTimeout(waitForFeed, 800);
        }
    };
    waitForFeed();
    return [];
}"""
