"""Archie — AI Solution Architect Debate (Gradio Web UI)

Entry point for the Gradio app. Run with:
    cd archie
    python app.py

Or deploy to Hugging Face Spaces.
"""

import sys
from pathlib import Path

# Ensure archie/ is on sys.path and .env is loaded before project imports
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from ui.layout import build_layout


def main() -> None:
    app, theme, css, js = build_layout()  # wire_events is called inside build_layout
    app.queue()
    # Gradio 6.0+ accepts theme, css, and js in launch()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=theme,
        css=css,
        js=js,
    )


if __name__ == "__main__":
    main()
