"""HTML builders for dynamic Gradio components."""

from __future__ import annotations

from ui.themes import AGENT_COLORS, AGENT_ICONS


def build_status_html(message: str, agent: str = "") -> str:
    """Build a status bar HTML string."""
    color = AGENT_COLORS.get(agent, "#94a3b8")
    icon = AGENT_ICONS.get(agent, "⏳")
    return (
        f'<div style="padding: 8px 16px; color: {color}; font-size: 14px; '
        f'border-left: 3px solid {color}; background: rgba(0,0,0,0.2); border-radius: 4px;">'
        f'{icon} {message}</div>'
    )


def build_event_banner(event: str) -> str:
    """Build a dramatic event banner HTML."""
    style_map = {
        "SCORE SHIFT!": ("🔥", "#ef4444", "rgba(239,68,68,0.1)"),
        "LEAD CHANGE!": ("🔄", "#ef4444", "rgba(239,68,68,0.15)"),
        "ADVERSARY CONCEDES!": ("🤝", "#22c55e", "rgba(34,197,94,0.1)"),
        "EXTRA ROUND GRANTED!": ("⚡", "#eab308", "rgba(234,179,8,0.1)"),
        "PROPOSER PIVOTS STRATEGY!": ("🔀", "#3b82f6", "rgba(59,130,246,0.1)"),
        "DEEP DIVE REQUESTED!": ("🔬", "#06b6d4", "rgba(6,182,212,0.1)"),
        "NEAR TIE!": ("⚖️", "#eab308", "rgba(234,179,8,0.1)"),
    }
    icon, color, bg = style_map.get(event, ("📢", "#94a3b8", "rgba(148,163,184,0.1)"))
    return (
        f'<div style="text-align: center; padding: 12px; margin: 8px 0; '
        f'background: {bg}; border: 1px solid {color}; border-radius: 8px; '
        f'font-weight: bold; font-size: 16px; color: {color}; letter-spacing: 1px;">'
        f'{icon} {event} {icon}</div>'
    )


def build_scoreboard_html(score_history: list[dict]) -> str:
    """Build an HTML scoreboard table."""
    if not score_history:
        return '<div style="color: #64748b; text-align: center; padding: 20px;">No scores yet</div>'

    rows = ""
    for i, entry in enumerate(score_history):
        # Trend
        if i == 0:
            trend = "—"
        else:
            prev_delta = score_history[i - 1]["delta"]
            curr_delta = entry["delta"]
            if curr_delta > prev_delta + 0.1:
                trend = f'<span style="color: {AGENT_COLORS["proposer"]}">▲ P</span>'
            elif curr_delta < prev_delta - 0.1:
                trend = f'<span style="color: {AGENT_COLORS["adversary"]}">▲ A</span>'
            else:
                trend = '<span style="color: #eab308">=</span>'

        # Delta color
        delta = entry["delta"]
        if delta > 0:
            delta_html = f'<span style="color: {AGENT_COLORS["proposer"]}">{delta:+.2f}</span>'
        elif delta < 0:
            delta_html = f'<span style="color: {AGENT_COLORS["adversary"]}">{delta:+.2f}</span>'
        else:
            delta_html = f'{delta:+.2f}'

        rows += (
            f'<tr style="border-bottom: 1px solid rgba(100,100,100,0.3);">'
            f'<td style="padding: 6px 10px; text-align: center;">{entry["round"]}</td>'
            f'<td style="padding: 6px 10px; text-align: center; color: {AGENT_COLORS["proposer"]}">{entry["proposer_score"]:.2f}</td>'
            f'<td style="padding: 6px 10px; text-align: center; color: {AGENT_COLORS["adversary"]}">{entry["adversary_score"]:.2f}</td>'
            f'<td style="padding: 6px 10px; text-align: center;">{delta_html}</td>'
            f'<td style="padding: 6px 10px; text-align: center;">{trend}</td>'
            f'</tr>'
        )

    return (
        f'<table style="width: 100%; border-collapse: collapse; font-size: 13px; color: #e2e8f0;">'
        f'<thead><tr style="border-bottom: 2px solid rgba(100,100,100,0.5);">'
        f'<th style="padding: 8px 10px; text-align: center; color: #94a3b8;">R</th>'
        f'<th style="padding: 8px 10px; text-align: center; color: {AGENT_COLORS["proposer"]}">P</th>'
        f'<th style="padding: 8px 10px; text-align: center; color: {AGENT_COLORS["adversary"]}">A</th>'
        f'<th style="padding: 8px 10px; text-align: center; color: #eab308;">Δ</th>'
        f'<th style="padding: 8px 10px; text-align: center;">Trend</th>'
        f'</tr></thead>'
        f'<tbody>{rows}</tbody></table>'
    )


def build_rubric_html(proposer_scores: dict, adversary_scores: dict) -> str:
    """Build an HTML rubric breakdown table with gap highlighting."""
    if not proposer_scores and not adversary_scores:
        return '<div style="color: #64748b; text-align: center; padding: 20px;">No rubric data yet</div>'

    dimensions = [
        ("constraint_adherence", "Constraints"),
        ("technical_feasibility", "Feasibility"),
        ("operational_complexity", "Operations"),
        ("scalability_fit", "Scalability"),
        ("evidence_quality", "Evidence"),
        ("cost_efficiency", "Cost"),
    ]

    rows = ""
    for dim_key, dim_label in dimensions:
        p_val = proposer_scores.get(dim_key, 0)
        a_val = adversary_scores.get(dim_key, 0)
        gap = abs(p_val - a_val)

        # Highlight large gaps
        row_bg = "rgba(234,179,8,0.08)" if gap > 0.5 else "transparent"
        gap_badge = ""
        if gap > 0.5:
            leader = "P" if p_val > a_val else "A"
            gap_badge = f' <span style="color: #eab308; font-size: 10px;">({leader} +{gap:.1f})</span>'

        rows += (
            f'<tr style="border-bottom: 1px solid rgba(100,100,100,0.2); background: {row_bg};">'
            f'<td style="padding: 5px 8px; color: #94a3b8; font-size: 12px;">{dim_label}{gap_badge}</td>'
            f'<td style="padding: 5px 8px; text-align: center; color: {AGENT_COLORS["proposer"]}; font-size: 13px;">{p_val:.1f}</td>'
            f'<td style="padding: 5px 8px; text-align: center; color: {AGENT_COLORS["adversary"]}; font-size: 13px;">{a_val:.1f}</td>'
            f'</tr>'
        )

    return (
        f'<table style="width: 100%; border-collapse: collapse; color: #e2e8f0;">'
        f'<thead><tr style="border-bottom: 2px solid rgba(100,100,100,0.5);">'
        f'<th style="padding: 6px 8px; text-align: left; color: #64748b; font-size: 11px;">Dimension</th>'
        f'<th style="padding: 6px 8px; text-align: center; color: {AGENT_COLORS["proposer"]}; font-size: 11px;">P</th>'
        f'<th style="padding: 6px 8px; text-align: center; color: {AGENT_COLORS["adversary"]}; font-size: 11px;">A</th>'
        f'</tr></thead>'
        f'<tbody>{rows}</tbody></table>'
    )


def build_momentum_html(momentum: dict) -> str:
    """Build momentum bar visualization as HTML."""
    if not momentum:
        return ""

    def _bar(value: float, color: str, label: str) -> str:
        # Clamp value for display
        clamped = max(-1.0, min(1.0, value))
        width = int(abs(clamped) * 100)
        sign = "+" if value >= 0 else ""
        return (
            f'<div style="display: flex; align-items: center; gap: 8px; margin: 4px 0;">'
            f'<span style="color: {color}; font-size: 12px; min-width: 20px;">{label}</span>'
            f'<div style="flex: 1; height: 8px; background: rgba(100,100,100,0.2); border-radius: 4px; overflow: hidden;">'
            f'<div style="width: {width}%; height: 100%; background: {color}; border-radius: 4px; '
            f'opacity: 0.7;"></div></div>'
            f'<span style="color: #94a3b8; font-size: 11px; min-width: 40px;">{sign}{value:.2f}</span>'
            f'</div>'
        )

    return (
        f'<div style="padding: 4px 0;">'
        f'{_bar(momentum.get("proposer", 0), AGENT_COLORS["proposer"], "P")}'
        f'{_bar(momentum.get("adversary", 0), AGENT_COLORS["adversary"], "A")}'
        f'</div>'
    )


def build_search_tracker_html(search_metadata: list[dict]) -> str:
    """Build search activity summary HTML."""
    if not search_metadata:
        return '<div style="color: #64748b; font-size: 12px;">No searches yet</div>'

    p_queries = sum(len(m.get("queries", [])) for m in search_metadata if m.get("agent") == "proposer")
    p_sources = sum(m.get("source_count", 0) for m in search_metadata if m.get("agent") == "proposer")
    a_queries = sum(len(m.get("queries", [])) for m in search_metadata if m.get("agent") == "adversary")
    a_sources = sum(m.get("source_count", 0) for m in search_metadata if m.get("agent") == "adversary")

    return (
        f'<div style="font-size: 12px; color: #94a3b8;">'
        f'<div style="color: {AGENT_COLORS["proposer"]}">P: {p_queries}q, {p_sources} sources</div>'
        f'<div style="color: {AGENT_COLORS["adversary"]}">A: {a_queries}q, {a_sources} sources</div>'
        f'</div>'
    )


def build_memory_display(memory: dict) -> str:
    """Build memory display for welcome screen."""
    parts = []
    if memory.get("cloud_providers"):
        parts.append(f"Cloud: {', '.join(memory['cloud_providers'])}")
    if memory.get("deployment_env"):
        parts.append(f"Deployment: {', '.join(memory['deployment_env'])}")
    if memory.get("domain"):
        parts.append(f"Domain: {memory['domain']}")
    if memory.get("team_size"):
        parts.append(f"Team: {memory['team_size']}")

    if not parts:
        return '<div style="color: #64748b; font-size: 13px;">No prior memory loaded.</div>'

    return (
        f'<div style="color: #94a3b8; font-size: 13px; padding: 8px 12px; '
        f'background: rgba(100,100,100,0.1); border-radius: 6px; border-left: 3px solid #64748b;">'
        f'📂 {" | ".join(parts)}</div>'
    )


def build_memory_diff_html(diff: dict) -> str:
    """Build memory diff display showing what was learned."""
    if not diff:
        return ""

    added = diff.get("added", {})
    updated = diff.get("updated", {})

    if not added and not updated:
        return '<div style="color: #64748b; font-size: 12px;">No new preferences learned.</div>'

    lines = []
    for field, items in added.items():
        for item in items:
            lines.append(f'<div style="color: #22c55e; font-size: 12px;">+ {field}: {item}</div>')
    for field, value in updated.items():
        lines.append(f'<div style="color: #eab308; font-size: 12px;">~ {field}: {value}</div>')

    return f'<div style="padding: 4px 0;">{"".join(lines)}</div>'
