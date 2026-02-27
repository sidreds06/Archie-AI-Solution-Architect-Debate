"""Plotly chart builders for Archie smart UI."""

from __future__ import annotations

import plotly.graph_objects as go

from ui.themes import AGENT_COLORS


def build_score_trend_chart(score_history: list[dict]) -> go.Figure:
    """Line chart showing proposer vs adversary scores across rounds."""
    if not score_history:
        return _empty_chart("Score Trend", "No data yet")

    rounds = [s["round"] for s in score_history]
    p_scores = [s["proposer_score"] for s in score_history]
    a_scores = [s["adversary_score"] for s in score_history]

    fig = go.Figure()

    # Shaded region between the two lines
    fig.add_trace(go.Scatter(
        x=rounds + rounds[::-1],
        y=p_scores + a_scores[::-1],
        fill="toself",
        fillcolor="rgba(100, 100, 100, 0.1)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False,
        hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(
        x=rounds, y=p_scores,
        mode="lines+markers",
        name="Proposer",
        line=dict(color=AGENT_COLORS["proposer"], width=3),
        marker=dict(size=10),
    ))

    fig.add_trace(go.Scatter(
        x=rounds, y=a_scores,
        mode="lines+markers",
        name="Adversary",
        line=dict(color=AGENT_COLORS["adversary"], width=3),
        marker=dict(size=10),
    ))

    fig.update_layout(
        title=dict(text="Score Trend", font=dict(size=14, color="#e2e8f0")),
        xaxis=dict(
            title="Round", dtick=1,
            gridcolor="rgba(100,100,100,0.3)",
            color="#94a3b8",
        ),
        yaxis=dict(
            title="Score", range=[1.0, 5.0],
            gridcolor="rgba(100,100,100,0.3)",
            color="#94a3b8",
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color="#e2e8f0")),
        margin=dict(l=40, r=20, t=40, b=30),
        height=250,
    )

    return fig


def build_rubric_radar_chart(
    proposer_scores: dict, adversary_scores: dict
) -> go.Figure:
    """Radar/spider chart comparing proposer vs adversary across 6 dimensions."""
    dimensions = [
        "constraint_adherence",
        "technical_feasibility",
        "operational_complexity",
        "scalability_fit",
        "evidence_quality",
        "cost_efficiency",
    ]
    labels = [d.replace("_", " ").title() for d in dimensions]

    p_values = [proposer_scores.get(d, 0) for d in dimensions]
    a_values = [adversary_scores.get(d, 0) for d in dimensions]

    # Close the polygon
    p_values.append(p_values[0])
    a_values.append(a_values[0])
    labels_closed = labels + [labels[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=p_values,
        theta=labels_closed,
        fill="toself",
        name="Proposer",
        line=dict(color=AGENT_COLORS["proposer"], width=2),
        fillcolor=f"rgba(59, 130, 246, 0.15)",
    ))

    fig.add_trace(go.Scatterpolar(
        r=a_values,
        theta=labels_closed,
        fill="toself",
        name="Adversary",
        line=dict(color=AGENT_COLORS["adversary"], width=2),
        fillcolor=f"rgba(34, 197, 94, 0.15)",
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, 5],
                gridcolor="rgba(100,100,100,0.3)",
                color="#94a3b8",
            ),
            angularaxis=dict(color="#94a3b8"),
            bgcolor="rgba(0,0,0,0)",
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color="#e2e8f0")),
        margin=dict(l=60, r=60, t=30, b=30),
        height=300,
        title=dict(text="Rubric Comparison", font=dict(size=14, color="#e2e8f0")),
    )

    return fig


def build_momentum_chart(score_history: list[dict]) -> go.Figure:
    """Bar chart showing momentum (score delta change) per round."""
    if len(score_history) < 2:
        return _empty_chart("Momentum", "Need 2+ rounds")

    rounds = []
    p_momentum = []
    a_momentum = []

    for i in range(1, len(score_history)):
        prev = score_history[i - 1]
        curr = score_history[i]
        rounds.append(curr["round"])
        p_momentum.append(curr["proposer_score"] - prev["proposer_score"])
        a_momentum.append(curr["adversary_score"] - prev["adversary_score"])

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=rounds, y=p_momentum,
        name="Proposer",
        marker_color=AGENT_COLORS["proposer"],
        opacity=0.8,
    ))

    fig.add_trace(go.Bar(
        x=rounds, y=a_momentum,
        name="Adversary",
        marker_color=AGENT_COLORS["adversary"],
        opacity=0.8,
    ))

    fig.update_layout(
        title=dict(text="Momentum", font=dict(size=14, color="#e2e8f0")),
        xaxis=dict(title="Round", dtick=1, color="#94a3b8", gridcolor="rgba(100,100,100,0.3)"),
        yaxis=dict(title="Score Change", color="#94a3b8", gridcolor="rgba(100,100,100,0.3)"),
        barmode="group",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color="#e2e8f0")),
        margin=dict(l=40, r=20, t=40, b=30),
        height=200,
    )

    return fig


def _empty_chart(title: str, message: str) -> go.Figure:
    """Return an empty placeholder chart."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=14, color="#64748b"),
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#e2e8f0")),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=200,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig
