"""Gradio event handlers — connect UI events to the core engine.

Debate-round handlers are generator functions that yield gr.update() tuples
so Gradio streams real-time token-by-token updates to the browser.
Non-debate handlers (start, export) remain synchronous.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import gradio as gr

from config import ADVERSARY_MODEL, PROPOSER_MODEL
from ui.charts import build_rubric_radar_chart, build_score_trend_chart
from ui.components.html_builders import (
    build_event_banner,
    build_memory_diff_html,
    build_memory_display,
    build_momentum_html,
    build_rubric_html,
    build_scoreboard_html,
    build_search_tracker_html,
    build_status_html,
)
from ui.export import export_json, export_markdown
from ui.themes import AGENT_COLORS, AGENT_ICONS

from markdown_it import MarkdownIt as _MarkdownIt
_md = _MarkdownIt()  # shared markdown renderer for agent panels


# ---------------------------------------------------------------------------
# HTML panel builders (unchanged)
# ---------------------------------------------------------------------------

def _build_collapsed_panel(
    agent: str,
    model: str,
    round_num: int,
    solution: str,
    score: float | None = None,
    sources: int = 0,
    is_deep_dive: bool = False,
) -> str:
    """Build an HTML <details> panel for a completed agent turn."""
    icon = AGENT_ICONS.get(agent, "📝")
    color = AGENT_COLORS.get(agent, "#94a3b8")
    label = agent.upper()
    css_class = "deep-dive" if is_deep_dive else agent

    summary_text = solution.split("\n")[0][:120]
    if len(solution.split("\n")[0]) > 120:
        summary_text += "..."

    score_badge = f" — {score:.2f}" if score is not None else ""
    dd_badge = " [DEEP DIVE]" if is_deep_dive else ""
    source_badge = f" — {sources} sources" if sources else ""

    rendered_html = _md.render(solution)

    return (
        f'<details class="agent-panel {css_class}">'
        f'<summary style="color: {color};">'
        f'{icon} {label} ({model}) — R{round_num}{score_badge}{dd_badge}{source_badge}'
        f'</summary>'
        f'<div class="panel-content">{rendered_html}</div>'
        f'</details>'
    )


def build_moderator_card_html(
    round_num: int,
    p_score: float,
    a_score: float,
    decision: str,
    reasoning: str,
) -> str:
    """Build a compact moderator card HTML."""
    delta = p_score - a_score
    leader = "P leads" if delta > 0 else "A leads" if delta < 0 else "Tied"
    delta_color = AGENT_COLORS["proposer"] if delta > 0 else AGENT_COLORS["adversary"]

    return (
        f'<div class="moderator-card">'
        f'<div style="display: flex; justify-content: space-between; align-items: center;">'
        f'<span>🟣 <strong>MODERATOR</strong> — R{round_num} — {decision.upper()}</span>'
        f'<span style="color: {delta_color}; font-size: 12px;">Δ {delta:+.2f} ({leader})</span>'
        f'</div>'
        f'<div style="display: flex; gap: 24px; margin-top: 6px; font-size: 13px;">'
        f'<span style="color: {AGENT_COLORS["proposer"]}">P: {p_score:.2f}</span>'
        f'<span style="color: {AGENT_COLORS["adversary"]}">A: {a_score:.2f}</span>'
        f'</div>'
        f'<div style="color: #94a3b8; font-size: 12px; margin-top: 6px;">{reasoning[:200]}</div>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Event wiring (unchanged)
# ---------------------------------------------------------------------------

def wire_events(app: gr.Blocks) -> None:
    """Wire all Gradio events to handler functions."""
    c = app.components_map

    # ---- Start Debate ----
    c["start_btn"].click(
        fn=handle_start_debate,
        inputs=[c["problem_input"], c["rounds_input"], c["engine_state"]],
        outputs=[
            c["engine_state"],
            c["welcome_phase"],
            c["interview_phase"],
            c["interview_chat"],
            c["interview_progress"],
            c["memory_display"],
            c["status_bar"],
        ],
    )

    # ---- Interview Submit ----
    c["interview_submit"].click(
        fn=handle_interview_answer,
        inputs=[c["interview_input"], c["engine_state"], c["interview_chat"]],
        outputs=[
            c["engine_state"],
            c["interview_chat"],
            c["interview_input"],
            c["interview_progress"],
            c["interview_phase"],
            c["debate_phase"],
            c["status_bar"],
            # Debate outputs
            c["debate_header"],
            c["debate_feed"],
            c["streaming_accordion"],
            c["streaming_content"],
            c["moderator_card"],
            c["scoreboard_html"],
            c["rubric_html"],
            c["momentum_html"],
            c["events_html"],
            c["search_html"],
            c["debate_status"],
            c["event_banner"],
            c["hitl_section"],
            c["hitl_context"],
            c["checkpoint_section"],
            c["checkpoint_info"],
            c["score_trend_chart"],
            # Verdict outputs
            c["verdict_phase"],
            c["verdict_scores"],
            c["verdict_brief"],
            c["verdict_radar"],
            c["verdict_trend"],
            c["verdict_stats"],
            c["verdict_memory_diff"],
        ],
    )

    # ---- Interview Skip ----
    c["interview_skip"].click(
        fn=handle_interview_skip,
        inputs=[c["engine_state"], c["interview_chat"]],
        outputs=[
            c["engine_state"],
            c["interview_chat"],
            c["interview_input"],
            c["interview_progress"],
            c["interview_phase"],
            c["debate_phase"],
            c["status_bar"],
            c["debate_header"],
            c["debate_feed"],
            c["streaming_accordion"],
            c["streaming_content"],
            c["moderator_card"],
            c["scoreboard_html"],
            c["rubric_html"],
            c["momentum_html"],
            c["events_html"],
            c["search_html"],
            c["debate_status"],
            c["event_banner"],
            c["hitl_section"],
            c["hitl_context"],
            c["checkpoint_section"],
            c["checkpoint_info"],
            c["score_trend_chart"],
            c["verdict_phase"],
            c["verdict_scores"],
            c["verdict_brief"],
            c["verdict_radar"],
            c["verdict_trend"],
            c["verdict_stats"],
            c["verdict_memory_diff"],
        ],
    )

    # ---- Continue (round checkpoint) ----
    c["continue_btn"].click(
        fn=handle_continue_round,
        inputs=[c["direction_input"], c["engine_state"], c["debate_feed"]],
        outputs=[
            c["engine_state"],
            c["debate_header"],
            c["debate_feed"],
            c["streaming_accordion"],
            c["streaming_content"],
            c["moderator_card"],
            c["scoreboard_html"],
            c["rubric_html"],
            c["momentum_html"],
            c["events_html"],
            c["search_html"],
            c["debate_status"],
            c["event_banner"],
            c["hitl_section"],
            c["hitl_context"],
            c["checkpoint_section"],
            c["checkpoint_info"],
            c["direction_input"],
            c["score_trend_chart"],
            c["status_bar"],
            c["verdict_phase"],
            c["debate_phase"],
            c["verdict_scores"],
            c["verdict_brief"],
            c["verdict_radar"],
            c["verdict_trend"],
            c["verdict_stats"],
            c["verdict_memory_diff"],
        ],
    )

    # ---- HITL Submit ----
    c["hitl_submit"].click(
        fn=handle_hitl_answer,
        inputs=[c["hitl_input"], c["engine_state"], c["debate_feed"]],
        outputs=[
            c["engine_state"],
            c["debate_feed"],
            c["hitl_section"],
            c["hitl_input"],
            c["scoreboard_html"],
            c["rubric_html"],
            c["momentum_html"],
            c["events_html"],
            c["search_html"],
            c["debate_status"],
            c["event_banner"],
            c["checkpoint_section"],
            c["checkpoint_info"],
            c["score_trend_chart"],
            c["status_bar"],
            c["hitl_context"],
            c["verdict_phase"],
            c["debate_phase"],
            c["verdict_scores"],
            c["verdict_brief"],
            c["verdict_radar"],
            c["verdict_trend"],
            c["verdict_stats"],
            c["verdict_memory_diff"],
        ],
    )

    # ---- HITL Skip ----
    c["hitl_skip"].click(
        fn=handle_hitl_skip,
        inputs=[c["engine_state"], c["debate_feed"]],
        outputs=[
            c["engine_state"],
            c["debate_feed"],
            c["hitl_section"],
            c["hitl_input"],
            c["scoreboard_html"],
            c["rubric_html"],
            c["momentum_html"],
            c["events_html"],
            c["search_html"],
            c["debate_status"],
            c["event_banner"],
            c["checkpoint_section"],
            c["checkpoint_info"],
            c["score_trend_chart"],
            c["status_bar"],
            c["hitl_context"],
            c["verdict_phase"],
            c["debate_phase"],
            c["verdict_scores"],
            c["verdict_brief"],
            c["verdict_radar"],
            c["verdict_trend"],
            c["verdict_stats"],
            c["verdict_memory_diff"],
        ],
    )

    # ---- Export ----
    c["export_md_btn"].click(
        fn=handle_export_md,
        inputs=[c["engine_state"]],
        outputs=[c["export_file"]],
    )
    c["export_json_btn"].click(
        fn=handle_export_json,
        inputs=[c["engine_state"]],
        outputs=[c["export_file"]],
    )


# ---------------------------------------------------------------------------
# Handler functions
# ---------------------------------------------------------------------------

def handle_start_debate(problem: str, rounds: int, engine_state):
    """Handle 'Start Debate' button click (synchronous — just loads first question)."""
    if not problem.strip():
        raise gr.Error("Please describe your architecture problem.")

    from core.engine import DebateEngine

    engine = DebateEngine(problem.strip(), max_rounds=int(rounds))
    memory = engine.load_memory()

    first_question = None
    for event in engine.start_interview():
        if event["type"] == "interview_question":
            first_question = event["question"]

    chat_history = []
    if first_question:
        chat_history = [{"role": "assistant", "content": f"**Q1:** {first_question}"}]

    return (
        engine,
        gr.update(visible=False),       # hide welcome
        gr.update(visible=True),         # show interview
        chat_history,
        "Question 1 of ~5",
        build_memory_display(memory),
        build_status_html("Interview started", "interviewer"),
    )


def handle_interview_answer(answer: str, engine, chat_history: list):
    """Handle interview answer. Generator: yields updates, streams debate if transitioning."""
    if engine is None:
        raise gr.Error("No active debate. Please start a new one.")

    chat_history = list(chat_history) + [{"role": "user", "content": answer}]

    next_question = None
    interview_done = False

    for event in engine.submit_interview_answer(answer):
        if event["type"] == "interview_question":
            next_question = event["question"]
        elif event["type"] == "interview_done":
            interview_done = True

    if next_question:
        q_num = sum(1 for m in chat_history if m["role"] == "assistant") + 1
        chat_history = chat_history + [
            {"role": "assistant", "content": f"**Q{q_num}:** {next_question}"}
        ]
        _no_op = gr.update()
        yield (
            engine, chat_history, "",
            f"Question {q_num} of ~5",
            gr.update(visible=True),    # interview_phase
            gr.update(visible=False),   # debate_phase
            build_status_html("Interview in progress", "interviewer"),
            *([_no_op] * 17),           # debate outputs — no change
            *([_no_op] * 7),            # verdict outputs — no change
        )
        return

    # Interview done — transition to debate with streaming
    yield from _transition_to_debate_gen(engine, chat_history)


def handle_interview_skip(engine, chat_history: list):
    """Skip remaining questions and start debate. Generator."""
    if engine is None:
        raise gr.Error("No active debate.")

    engine.skip_interview()
    chat_history = list(chat_history) + [
        {"role": "assistant", "content": "*Interview skipped. Starting debate...*"}
    ]
    yield from _transition_to_debate_gen(engine, chat_history)


def handle_continue_round(direction: str, engine, debate_feed_html: str):
    """Handle 'Continue' at round checkpoint. Generator with streaming."""
    if engine is None:
        raise gr.Error("No active debate.")

    engine.increment_round()
    for updates in _run_debate_round_gen(engine, direction=direction, prev_feed=debate_feed_html):
        yield _continue_tuple(engine, updates)


def handle_hitl_answer(answer: str, engine, debate_feed_html: str):
    """Handle HITL answer submission. Generator."""
    if engine is None:
        raise gr.Error("No active debate.")

    for updates in _run_hitl_rescore_gen(engine, answer, prev_feed=debate_feed_html):
        yield _hitl_tuple(engine, updates)


def handle_hitl_skip(engine, debate_feed_html: str):
    """Skip HITL and continue with a neutral answer. Generator."""
    yield from handle_hitl_answer("No preference — continue the debate.", engine, debate_feed_html)


def handle_export_md(engine):
    """Export debate as Markdown."""
    if engine is None:
        raise gr.Error("No debate to export.")
    content = export_markdown(engine.state)
    path = Path(tempfile.mktemp(suffix=".md"))
    path.write_text(content, encoding="utf-8")
    return gr.update(value=str(path), visible=True)


def handle_export_json(engine):
    """Export debate as JSON."""
    if engine is None:
        raise gr.Error("No debate to export.")
    content = export_json(engine.state)
    path = Path(tempfile.mktemp(suffix=".json"))
    path.write_text(content, encoding="utf-8")
    return gr.update(value=str(path), visible=True)


# ---------------------------------------------------------------------------
# Output tuple builders
# Each builder maps an update dict to the fixed-length tuple required by the
# corresponding Gradio outputs list.  Missing keys fall back to gr.update()
# (no-op — preserves the current component value in the browser).
# ---------------------------------------------------------------------------

def _interview_to_debate_tuple(engine, chat_history: list, updates: dict) -> tuple:
    """31-element tuple for interview_submit / interview_skip outputs."""
    _n = gr.update()
    return (
        engine,                                                             # 1  engine_state
        chat_history,                                                       # 2  interview_chat
        "",                                                                 # 3  interview_input
        "Interview complete!",                                              # 4  interview_progress
        gr.update(visible=False),                                           # 5  interview_phase
        updates.get("debate_phase_visible", gr.update(visible=True)),       # 6  debate_phase
        updates.get("status_bar", _n),                                      # 7  status_bar
        updates.get("debate_header", _n),                                   # 8  debate_header
        updates.get("debate_feed", _n),                                     # 9  debate_feed
        updates.get("streaming_accordion", _n),                             # 10 streaming_accordion
        updates.get("streaming_content", _n),                               # 11 streaming_content
        updates.get("moderator_card", _n),                                  # 12 moderator_card
        updates.get("scoreboard_html", _n),                                 # 13 scoreboard_html
        updates.get("rubric_html", _n),                                     # 14 rubric_html
        updates.get("momentum_html", _n),                                   # 15 momentum_html
        updates.get("events_html", _n),                                     # 16 events_html
        updates.get("search_html", _n),                                     # 17 search_html
        updates.get("debate_status", _n),                                   # 18 debate_status
        updates.get("event_banner", _n),                                    # 19 event_banner
        updates.get("hitl_section", _n),                                    # 20 hitl_section
        updates.get("hitl_context", _n),                                    # 21 hitl_context
        updates.get("checkpoint_section", _n),                              # 22 checkpoint_section
        updates.get("checkpoint_info", _n),                                 # 23 checkpoint_info
        updates.get("score_trend_chart", _n),                               # 24 score_trend_chart
        updates.get("verdict_phase", gr.update(visible=False)),             # 25 verdict_phase
        updates.get("verdict_scores", _n),                                  # 26 verdict_scores
        updates.get("verdict_brief", _n),                                   # 27 verdict_brief
        updates.get("verdict_radar", _n),                                   # 28 verdict_radar
        updates.get("verdict_trend", _n),                                   # 29 verdict_trend
        updates.get("verdict_stats", _n),                                   # 30 verdict_stats
        updates.get("verdict_memory_diff", _n),                             # 31 verdict_memory_diff
    )


def _continue_tuple(engine, updates: dict) -> tuple:
    """28-element tuple for continue_btn outputs."""
    _n = gr.update()
    return (
        engine,                                                             # 1  engine_state
        updates.get("debate_header", _n),                                   # 2  debate_header
        updates.get("debate_feed", _n),                                     # 3  debate_feed
        updates.get("streaming_accordion", _n),                             # 4  streaming_accordion
        updates.get("streaming_content", _n),                               # 5  streaming_content
        updates.get("moderator_card", _n),                                  # 6  moderator_card
        updates.get("scoreboard_html", _n),                                 # 7  scoreboard_html
        updates.get("rubric_html", _n),                                     # 8  rubric_html
        updates.get("momentum_html", _n),                                   # 9  momentum_html
        updates.get("events_html", _n),                                     # 10 events_html
        updates.get("search_html", _n),                                     # 11 search_html
        updates.get("debate_status", _n),                                   # 12 debate_status
        updates.get("event_banner", _n),                                    # 13 event_banner
        updates.get("hitl_section", _n),                                    # 14 hitl_section
        updates.get("hitl_context", _n),                                    # 15 hitl_context
        updates.get("checkpoint_section", _n),                              # 16 checkpoint_section
        updates.get("checkpoint_info", _n),                                 # 17 checkpoint_info
        updates.get("direction_input", _n),                                 # 18 direction_input
        updates.get("score_trend_chart", _n),                               # 19 score_trend_chart
        updates.get("status_bar", _n),                                      # 20 status_bar
        updates.get("verdict_phase", _n),                                   # 21 verdict_phase
        updates.get("debate_phase_visible", _n),                            # 22 debate_phase
        updates.get("verdict_scores", _n),                                  # 23 verdict_scores
        updates.get("verdict_brief", _n),                                   # 24 verdict_brief
        updates.get("verdict_radar", _n),                                   # 25 verdict_radar
        updates.get("verdict_trend", _n),                                   # 26 verdict_trend
        updates.get("verdict_stats", _n),                                   # 27 verdict_stats
        updates.get("verdict_memory_diff", _n),                             # 28 verdict_memory_diff
    )


def _hitl_tuple(engine, updates: dict) -> tuple:
    """24-element tuple for hitl_submit / hitl_skip outputs."""
    _n = gr.update()
    return (
        engine,                                                             # 1  engine_state
        updates.get("debate_feed", _n),                                     # 2  debate_feed
        updates.get("hitl_section", _n),                                    # 3  hitl_section
        "",                                                                 # 4  hitl_input (always clear)
        updates.get("scoreboard_html", _n),                                 # 5  scoreboard_html
        updates.get("rubric_html", _n),                                     # 6  rubric_html
        updates.get("momentum_html", _n),                                   # 7  momentum_html
        updates.get("events_html", _n),                                     # 8  events_html
        updates.get("search_html", _n),                                     # 9  search_html
        updates.get("debate_status", _n),                                   # 10 debate_status
        updates.get("event_banner", _n),                                    # 11 event_banner
        updates.get("checkpoint_section", _n),                              # 12 checkpoint_section
        updates.get("checkpoint_info", _n),                                 # 13 checkpoint_info
        updates.get("score_trend_chart", _n),                               # 14 score_trend_chart
        updates.get("status_bar", _n),                                      # 15 status_bar
        updates.get("hitl_context", _n),                                    # 16 hitl_context
        updates.get("verdict_phase", _n),                                   # 17 verdict_phase
        updates.get("debate_phase_visible", _n),                            # 18 debate_phase
        updates.get("verdict_scores", _n),                                  # 19 verdict_scores
        updates.get("verdict_brief", _n),                                   # 20 verdict_brief
        updates.get("verdict_radar", _n),                                   # 21 verdict_radar
        updates.get("verdict_trend", _n),                                   # 22 verdict_trend
        updates.get("verdict_stats", _n),                                   # 23 verdict_stats
        updates.get("verdict_memory_diff", _n),                             # 24 verdict_memory_diff
    )


# ---------------------------------------------------------------------------
# Core generator: interview → debate transition
# ---------------------------------------------------------------------------

def _transition_to_debate_gen(engine, chat_history: list):
    """Generator: run first round and yield interview→debate transition tuples (31 outputs)."""
    for updates in _run_debate_round_gen(engine):
        yield _interview_to_debate_tuple(engine, chat_history, updates)


# ---------------------------------------------------------------------------
# Core generator: debate round
# ---------------------------------------------------------------------------

def _run_debate_round_gen(engine, direction: str = "", prev_feed: str = ""):
    """Generator that runs one debate round and yields update dicts.

    Intermediate yields carry only the keys that changed so callers can
    merge with gr.update() no-ops for unchanged components.
    The final yield carries the complete sidebar, routing, and verdict state.
    """
    feed_html = prev_feed or ""
    current_agent = ""
    current_text = ""
    current_model = ""
    current_round = engine.current_round
    current_sources = 0
    last_score_event = None
    hitl_requested = False
    hitl_question = ""
    events_html_parts: list[str] = []
    route = "continue"

    for event in engine.run_round(direction=direction):
        etype = event["type"]

        if etype == "stream_start":
            current_agent = event.get("agent", "")
            current_model = event.get("model", "")
            current_round = event.get("round", current_round)
            current_text = ""
            current_sources = 0
            icon = AGENT_ICONS.get(current_agent, "📝")
            color = AGENT_COLORS.get(current_agent, "#94a3b8")
            yield {
                "streaming_accordion": gr.update(
                    visible=True,
                    label=f"{icon} {current_agent.upper()} ({current_model}) — Round {current_round}",
                ),
                "streaming_content": "",
                "debate_status": build_status_html(
                    f"{current_agent.title()} is generating a response...", current_agent
                ),
                "status_bar": build_status_html(
                    f"Round {current_round}: {current_agent.title()} thinking...", current_agent
                ),
            }

        elif etype == "token":
            current_text += event.get("token", "")
            yield {"streaming_content": current_text}

        elif etype == "stream_end":
            current_sources = event.get("sources", 0)
            solution = event.get("solution", current_text)
            is_dd = event.get("is_deep_dive", False)
            feed_html += _build_collapsed_panel(
                agent=current_agent,
                model=current_model,
                round_num=current_round,
                solution=solution,
                score=None,
                sources=current_sources,
                is_deep_dive=is_dd,
            )
            yield {
                "debate_feed": feed_html,
                "streaming_accordion": gr.update(visible=False),
                "streaming_content": "",
                "debate_status": build_status_html(
                    f"{current_agent.title()} done", current_agent
                ),
            }

        elif etype == "search_done":
            current_sources = event.get("sources", 0)
            yield {
                "debate_status": build_status_html(
                    f"{current_agent.title()} gathered {current_sources} sources", current_agent
                ),
            }

        elif etype == "deep_dive_start":
            topic = event.get("topic", "")
            yield {
                "debate_status": build_status_html(
                    f"Deep dive: {topic[:60]}", "moderator"
                ),
            }

        elif etype == "score":
            last_score_event = event

        elif etype == "hitl_request":
            hitl_requested = True
            hitl_question = event.get("question", "")

        elif etype == "agent_request":
            banner = build_event_banner(
                f"{event.get('request_type', '').upper().replace('_', ' ')}!"
            )
            events_html_parts.append(banner)

        elif etype == "route":
            route = event.get("decision", "continue")

    # --- Build moderator card ---
    mod_card = ""
    if last_score_event:
        mod_card = build_moderator_card_html(
            round_num=last_score_event.get("round", current_round),
            p_score=last_score_event.get("proposer_score", 0),
            a_score=last_score_event.get("adversary_score", 0),
            decision=last_score_event.get("decision", ""),
            reasoning=last_score_event.get("reasoning", ""),
        )
        # mod_card is shown via the dedicated moderator_card component — don't duplicate in feed

    # --- Dramatic event banners ---
    state = engine.state
    for evt in state.get("dramatic_events", []):
        banner = build_event_banner(evt)
        feed_html += banner
        events_html_parts.append(banner)

    # --- Sidebar data ---
    score_history = state.get("score_history", [])
    momentum = state.get("momentum", {})
    last_verdict = state.get("final_verdict") or {}
    dim_scores_p = last_verdict.get("scores", {}).get("proposer", {})
    dim_scores_a = last_verdict.get("scores", {}).get("adversary", {})

    final_updates: dict = {
        "debate_header": f"## ⚔️ DEBATE — Round {current_round} of {engine.max_rounds}",
        "debate_feed": feed_html,
        "streaming_accordion": gr.update(visible=False),
        "streaming_content": "",
        # Fix: set both visible AND value so the card is never blank
        "moderator_card": gr.update(visible=bool(mod_card), value=mod_card),
        "scoreboard_html": build_scoreboard_html(score_history),
        "rubric_html": build_rubric_html(
            {k: v for k, v in dim_scores_p.items() if k != "weighted_total"},
            {k: v for k, v in dim_scores_a.items() if k != "weighted_total"},
        ),
        "momentum_html": build_momentum_html(momentum),
        "events_html": "".join(events_html_parts) if events_html_parts else "",
        "search_html": build_search_tracker_html(state.get("search_metadata", [])),
        "debate_status": build_status_html(
            f"Round {current_round} complete", "moderator"
        ),
        "event_banner": gr.update(visible=False),
        "score_trend_chart": gr.update(
            value=build_score_trend_chart(score_history),
            visible=bool(score_history),
        ),
        "status_bar": build_status_html(
            f"Round {current_round} of {engine.max_rounds} complete", "moderator"
        ),
        "direction_input": "",  # clear direction input after each round
    }

    # --- Route handling ---
    if hitl_requested:
        p_score = last_score_event.get("proposer_score", 0) if last_score_event else 0
        a_score = last_score_event.get("adversary_score", 0) if last_score_event else 0
        final_updates["hitl_section"] = gr.update(visible=True)
        final_updates["hitl_context"] = (
            f"**Scores:** Proposer {p_score:.2f} | Adversary {a_score:.2f}\n\n"
            f"{hitl_question}"
        )
        final_updates["checkpoint_section"] = gr.update(visible=False)
        final_updates["checkpoint_info"] = ""

    elif route == "verdict" or engine.is_debate_over:
        verdict_updates = _run_verdict(engine)
        final_updates.update(verdict_updates)
        final_updates["hitl_section"] = gr.update(visible=False)
        final_updates["hitl_context"] = ""
        final_updates["checkpoint_section"] = gr.update(visible=False)
        final_updates["checkpoint_info"] = ""

    else:
        final_updates["hitl_section"] = gr.update(visible=False)
        final_updates["hitl_context"] = ""
        final_updates["checkpoint_section"] = gr.update(visible=True)
        final_updates["checkpoint_info"] = (
            f"**Round {current_round} of {engine.max_rounds} complete.**\n\n"
            "Press Continue or type direction for the agents."
        )

    yield final_updates


# ---------------------------------------------------------------------------
# Core generator: HITL rescore
# ---------------------------------------------------------------------------

def _run_hitl_rescore_gen(engine, answer: str, prev_feed: str = ""):
    """Generator that rescores after HITL answer and yields update dicts."""
    feed_html = prev_feed
    hitl_requested = False
    hitl_question = ""
    route = "continue"

    # Immediate feedback: hide HITL section and show "rescoring" status
    yield {
        "hitl_section": gr.update(visible=False),
        "hitl_context": "",
        "debate_status": build_status_html("Re-scoring with your input...", "moderator"),
        "status_bar": build_status_html("Moderator re-scoring...", "moderator"),
    }

    for event in engine.run_hitl_rescore(answer):
        etype = event["type"]

        if etype == "hitl_rescore":
            old = event.get("old_scores", {})
            new = event.get("new_scores", {})
            old_p = old.get("proposer", {}).get("weighted_total", 0)
            old_a = old.get("adversary", {}).get("weighted_total", 0)
            new_p = new.get("proposer", {}).get("weighted_total", 0)
            new_a = new.get("adversary", {}).get("weighted_total", 0)

            p_change = new_p - old_p
            a_change = new_a - old_a
            p_arrow = "↑" if p_change > 0 else "↓" if p_change < 0 else "="
            a_arrow = "↑" if a_change > 0 else "↓" if a_change < 0 else "="

            rescore_html = (
                f'<div style="background: rgba(234,179,8,0.05); border: 1px solid rgba(234,179,8,0.3); '
                f'border-radius: 8px; padding: 10px; margin: 6px 0; font-size: 13px;">'
                f'<strong>HITL Re-score:</strong><br>'
                f'Proposer: {old_p:.2f} → {new_p:.2f} ({p_arrow} {p_change:+.2f})<br>'
                f'Adversary: {old_a:.2f} → {new_a:.2f} ({a_arrow} {a_change:+.2f})'
                f'</div>'
            )
            feed_html += rescore_html
            yield {"debate_feed": feed_html}

        elif etype == "hitl_request":
            hitl_requested = True
            hitl_question = event.get("question", "")

        elif etype == "route":
            route = event.get("decision", "continue")

    state = engine.state
    score_history = state.get("score_history", [])
    momentum = state.get("momentum", {})
    last_verdict = state.get("final_verdict") or {}
    dim_scores_p = last_verdict.get("scores", {}).get("proposer", {})
    dim_scores_a = last_verdict.get("scores", {}).get("adversary", {})

    final_updates: dict = {
        "debate_feed": feed_html,
        "scoreboard_html": build_scoreboard_html(score_history),
        "rubric_html": build_rubric_html(
            {k: v for k, v in dim_scores_p.items() if k != "weighted_total"},
            {k: v for k, v in dim_scores_a.items() if k != "weighted_total"},
        ),
        "momentum_html": build_momentum_html(momentum),
        "events_html": "",
        "search_html": build_search_tracker_html(state.get("search_metadata", [])),
        "debate_status": build_status_html("Re-scored with your input", "moderator"),
        "event_banner": gr.update(visible=False),
        "score_trend_chart": gr.update(
            value=build_score_trend_chart(score_history),
            visible=bool(score_history),
        ),
        "status_bar": build_status_html("Re-scored with HITL input", "moderator"),
    }

    if hitl_requested:
        final_updates["hitl_section"] = gr.update(visible=True)
        final_updates["hitl_context"] = hitl_question
        final_updates["checkpoint_section"] = gr.update(visible=False)
        final_updates["checkpoint_info"] = ""
    elif route == "verdict" or engine.is_debate_over:
        verdict_updates = _run_verdict(engine)
        final_updates.update(verdict_updates)
        final_updates["hitl_section"] = gr.update(visible=False)
        final_updates["hitl_context"] = ""
        final_updates["checkpoint_section"] = gr.update(visible=False)
        final_updates["checkpoint_info"] = ""
    else:
        final_updates["hitl_section"] = gr.update(visible=False)
        final_updates["hitl_context"] = ""
        final_updates["checkpoint_section"] = gr.update(visible=True)
        final_updates["checkpoint_info"] = (
            f"**Round {engine.current_round} of {engine.max_rounds} complete.**\n\n"
            "Press Continue or type direction."
        )

    yield final_updates


# ---------------------------------------------------------------------------
# Verdict (synchronous — returns a dict, called from within generators)
# ---------------------------------------------------------------------------

def _run_verdict(engine) -> dict:
    """Run the verdict phase and return a dict of UI updates."""
    brief = ""
    memory_diff = {}
    winner = "TBD"
    final_scores = {}

    for event in engine.run_verdict():
        etype = event["type"]
        if etype == "token":
            brief += event.get("token", "")
        elif etype == "verdict_done":
            brief = event.get("brief", brief)
            winner = event.get("winner", "TBD")
            memory_diff = event.get("memory_diff", {})
            final_scores = event.get("final_scores", {})

    state = engine.state
    score_history = state.get("score_history", [])
    stats = engine.get_session_stats()
    last_verdict = state.get("final_verdict") or {}
    dim_p = last_verdict.get("scores", {}).get("proposer", {})
    dim_a = last_verdict.get("scores", {}).get("adversary", {})

    duration = stats.get("duration_seconds", 0)
    mins = int(duration // 60)
    secs = int(duration % 60)

    p_final = final_scores.get("proposer_score", 0)
    a_final = final_scores.get("adversary_score", 0)

    return {
        "verdict_phase": gr.update(visible=True),
        "debate_phase_visible": gr.update(visible=False),
        "verdict_scores": (
            f"### Winner: **{winner.upper()}**\n\n"
            f"**Proposer:** {p_final:.2f}\n\n"
            f"**Adversary:** {a_final:.2f}"
        ),
        "verdict_brief": brief,
        "verdict_radar": build_rubric_radar_chart(
            {k: v for k, v in dim_p.items() if k != "weighted_total"},
            {k: v for k, v in dim_a.items() if k != "weighted_total"},
        ),
        "verdict_trend": build_score_trend_chart(score_history),
        "verdict_stats": (
            f"**Duration:** {mins}m {secs}s\n\n"
            f"**Rounds:** {stats.get('rounds_played', 0)}\n\n"
            f"**Searches:** {stats.get('total_searches', 0)}\n\n"
            f"**HITL:** {stats.get('hitl_count', 0)} questions\n\n"
            f"**Deep dives:** {stats.get('deep_dives', 0)}"
        ),
        "verdict_memory_diff": build_memory_diff_html(memory_diff),
    }
