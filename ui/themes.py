"""Custom Gradio theme for Archie."""

import gradio as gr


def build_theme() -> gr.themes.Base:
    """Dark theme with debate agent colors."""
    return gr.themes.Soft(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.green,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("Inter"),
        font_mono=gr.themes.GoogleFont("JetBrains Mono"),
    ).set(
        body_background_fill="*neutral_950",
        body_background_fill_dark="*neutral_950",
        block_background_fill="*neutral_900",
        block_background_fill_dark="*neutral_900",
        block_border_color="*neutral_700",
        block_label_text_color="*neutral_300",
        block_title_text_color="*neutral_100",
        input_background_fill="*neutral_800",
        input_background_fill_dark="*neutral_800",
        button_primary_background_fill="*primary_600",
        button_primary_text_color="white",
    )


# Agent color constants for use in HTML/CSS
AGENT_COLORS = {
    "proposer": "#3b82f6",   # blue-500
    "adversary": "#22c55e",  # green-500
    "moderator": "#a855f7",  # purple-500
    "interviewer": "#eab308", # yellow-500
    "deep_dive": "#06b6d4",  # cyan-500
    "verdict": "#22c55e",    # green-500
}

AGENT_LABELS = {
    "proposer": "PROPOSER",
    "adversary": "ADVERSARY",
    "moderator": "MODERATOR",
    "interviewer": "INTERVIEWER",
}

AGENT_ICONS = {
    "proposer": "🔵",
    "adversary": "🟢",
    "moderator": "🟣",
    "interviewer": "🟡",
}
