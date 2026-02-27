# Archie - AI Solution Architect Debate Agent

A multi-agent CLI and web app where specialized AI agents debate your architecture problem, score each other's solutions, and deliver a verdict — all powered by frontier models via OpenRouter.

## How It Works

You describe an architecture problem. Three AI agents take over:

- **Proposer** - a confident, battle-tested architect who defends a solution backed by web research
- **Adversary** - a ruthless, evidence-driven critic who hunts for flaws, failures, and better alternatives
- **Moderator** - a neutral judge who scores each round, detects when human input is needed, and decides when the debate is over

A fourth agent, the **Interviewer**, runs before the debate starts to ask clarifying questions - so the agents actually understand your constraints before arguing.


## Features

- **Pre-debate interview** - Gemini Flash generates 3–5 clarifying questions; answers are injected into all agent prompts
- **Live token streaming** - all agent responses stream token-by-token in real time
- **Evidence-backed arguments** - Proposer runs 2 best-practice searches; Adversary runs 5 failure/alternative searches via Tavily; all sources are cited inline
- **Live scoreboard** - Rich table after every round showing scores, delta, trend, and dramatic events (LEAD CHANGE!, SCORE SHIFT!, ADVERSARY CONCEDES!, etc.)
- **Smart HITL pauses** - the moderator automatically requests human input when scores are too close, agents make unstated assumptions, or a preference is unspecified
- **Dynamic graph** - agents can request deep dives on specific topics, extra rounds, or concede; the graph re-routes accordingly
- **Momentum injection** - losing agents get urgency in their prompts; winning agents get confidence
- **Persistent memory** - preferences are extracted after each session and merged additively into `memory.json` for use in future runs
- **Gradio web UI** - alternative to the CLI with live score trend charts and HTML export

## Agents and Models

| Agent | Model | Role |
|---|---|---|
| Proposer | `openai/gpt-5.2` | Confident architect defending a solution |
| Adversary | `anthropic/claude-sonnet-4-5` | Ruthless critic finding failures and alternatives |
| Moderator | `google/gemini-2.5-flash` | Neutral scorer and routing decision-maker |
| Interviewer | `google/gemini-2.5-flash` | Pre-debate clarification questions |

All models are accessed via [OpenRouter](https://openrouter.ai/) using the standard OpenAI-compatible API.

## Graph Flow

```
START
  ↓
load_memory          ← restore persistent user preferences
  ↓
interviewer          ← generate clarifying questions, build enriched_context
  ↓
proposer             ← solution + 2 web searches
  ↓
adversary            ← critique + 5 web searches
  ↓
moderator            ← score, apply termination rules, inline HITL if needed
  ↓
scoreboard           ← Rich table, detect dramatic events
  ↓
request_handler      ← process agent JSON action requests
  ↓
route_after_requests()
  ├── deep_dive_proposer → deep_dive_adversary → moderator
  ├── round_increment → proposer   (next round)
  └── verdict → END
```

## Moderator Termination Rules

Applied in order each round:

1. `round >= max_rounds` → end
2. Score delta `< 0.3` and `round >= 2` → HITL (contested)
3. Both agents assume an unstated constraint → HITL
4. Agents debate an unspecified user preference → HITL
5. Score delta `< 0.5` and `round >= 3` → end (converging)
6. One score `>= 4.0` and gap `>= 0.8` → end (clear winner)
7. Otherwise → continue

## Prerequisites

- Python 3.11+
- An [OpenRouter](https://openrouter.ai/) API key (covers all three models)
- A [Tavily](https://tavily.com/) API key (web search)

## Installation

```bash
git clone https://github.com/your-username/AI_Solution_Architect_Debate_Agent.git
cd AI_Solution_Architect_Debate_Agent/archie

python -m venv .venv

# Windows
.venv\Scripts\pip install -r requirements.txt

# macOS / Linux
.venv/bin/pip install -r requirements.txt
```

## Configuration

Create `archie/.env`:

```env
OPENROUTER_API_KEY=your_openrouter_key_here
TAVILY_API_KEY=your_tavily_key_here
```

## Usage

### CLI

```bash
cd archie

# Default (5 rounds)
.venv\Scripts\python main.py

# Custom round count
.venv\Scripts\python main.py --rounds 3
```

You will be prompted to describe your architecture problem. The debate begins immediately after the interview phase.

### Gradio Web UI

```bash
cd archie
.venv\Scripts\python app.py
```

Open [http://localhost:7860](http://localhost:7860) in your browser. The web UI provides real-time streaming, live score trend charts, and HTML export of the full debate transcript.

### Sanity Check (no API calls)

```bash
cd archie
.venv\Scripts\python -c "from graph import build_graph; build_graph(); print('OK')"
```

## Project Structure

```
archie/
├── main.py               # CLI entry point
├── app.py                # Gradio web UI entry point
├── graph.py              # LangGraph orchestration and conditional routing
├── state.py              # DebateState TypedDict
├── config.py             # All constants (models, thresholds, weights)
├── requirements.txt
├── .env                  # API keys (not committed)
├── nodes/
│   ├── interviewer.py    # Pre-debate Q&A
│   ├── proposer.py       # Solution generation + search
│   ├── adversary.py      # Critique + search
│   ├── moderator.py      # Scoring, routing, inline HITL
│   ├── scoreboard.py     # Rich scoreboard table
│   ├── deep_dive.py      # Focused sub-debate nodes
│   ├── request_handler.py
│   ├── round_increment.py
│   ├── verdict.py
│   └── loader.py
├── prompts/              # Agent system prompts
├── tools/
│   └── search.py         # Tavily search helpers
├── memory/
│   ├── manager.py        # Load / save / update with additive merge
│   └── memory.json       # Persistent user preferences
└── ui/                   # Gradio UI components
    ├── layout.py
    ├── handlers.py
    ├── streaming.py
    ├── themes.py
    ├── charts.py
    ├── dramatic.py
    ├── export.py
    └── components/
```

## Scoring Rubric

The moderator scores each solution on a 1–5 scale across six dimensions:

| Dimension | Weight |
|---|---|
| Constraint adherence | 25% |
| Technical feasibility | 20% |
| Operational complexity | 20% |
| Scalability fit | 15% |
| Evidence quality | 10% |
| Cost efficiency | 10% |

## Tech Stack

| Library | Purpose |
|---|---|
| [LangGraph](https://github.com/langchain-ai/langgraph) | Stateful multi-agent graph orchestration |
| [LangChain OpenAI](https://github.com/langchain-ai/langchain) | OpenRouter API integration |
| [Tavily Python](https://github.com/tavily-ai/tavily-python) | Web search for evidence gathering |
| [Rich](https://github.com/Textualize/rich) | Colored CLI output, streaming panels, live tables |
| [Gradio](https://github.com/gradio-app/gradio) | Web UI with real-time streaming |
| [Plotly](https://plotly.com/python/) | Score trend charts in the web UI |
| [python-dotenv](https://github.com/theskumar/python-dotenv) | Environment variable loading |


## Walkthrough

### 1. Startup & Pre-Debate Interview

![Startup and Requirements Interview](Images/_01_Img_Start.png)

Archie opens with a header panel showing the three models in play, then immediately hands off to the **Interviewer** agent. Before any debate begins, Gemini Flash generates 3–5 targeted clarifying questions based on your problem description. In this example — designing a production RAG system for a legal firm's document QA - the interviewer asks about acceptable retrieval latency, the technical level of end users, document update frequency, compliance requirements, and document format mix (PDFs, scanned images, Word docs). Your answers are assembled into an `enriched_context` block that is injected into every subsequent agent prompt, ensuring the debate is grounded in your actual constraints rather than generic assumptions.

---

### 2. Proposer - Round 1

![Proposer Round 1](Images/_02_Img_Proposer_Round_1.png)

The **Proposer** (GPT-5.2, shown in the blue panel) opens with a full architecture proposal. It first states the target properties derived from the interview - zero data egress, full auditability down to chunk IDs and page numbers, low hallucination tolerance (the system is allowed to say "I don't know"), and a p95 latency budget of under 10 seconds. It then lays out a concrete stack: **MinIO** for immutable raw document storage with separate `raw/`, `normalized/`, `chunks/`, and `audit/` buckets; **PostgreSQL** for metadata governance with per-chunk offsets and bounding boxes for OCR citation highlighting; and **Kubernetes** as the execution substrate. The proposer backs its design choices with reasoning specific to the legal context — citations in legal QA mean page-level evidence, not URLs.

---

### 3. Adversary - Round 1

![Adversary Round 1](Images/_03_Img_Advesary_Round_1.png)

The **Adversary** (Claude Sonnet, shown in the green panel) tears into the proposer's design immediately. It challenges the reranker choice - pointing out that `bge-reranker-large` runs 20–30 seconds for 200 documents on a single GPU, blowing through the 10-second latency SLA. It attacks the verification gate design, arguing that using the same LLM to both compose and verify answers is circular and does not catch confabulation from stale or jurisdictionally misapplied case law. Every claim is backed by numbered citations pulled from live web searches — real sources including HuggingFace discussions on reranker latency, an arXiv paper on HyperRAG KV-cache reuse, and Elastic's blog on ACL filtering degradation.

---

### 4. Evidence Sources, Moderator Scoring & Deep Dive

![Sources, Moderator and Scoreboard](Images/_04_Img_Sources_Moderator.png)

At the bottom of the adversary's response, all web sources are listed explicitly — this is the evidence layer that distinguishes Archie from a simple chatbot. The **Moderator** (Gemini Flash, magenta panel) then scores the round: **Proposer 3.48 / Adversary 4.08**, decision CONTINUE. Its commentary explains that the adversary's evidence-backed critique of the reranker and verification strategies outweighed the proposer's sound but incompletely stress-tested design.

Below the moderator, the live **scoreboard table** appears — Round, Proposer score (blue), Adversary score (green), Delta, and Trend column. The proposer then exercises the **dynamic graph** feature by requesting a deep dive: `ACL security trimming strategies in OpenSearch at 10M-page scale (group expansion vs denormalized fields vs per-matter indices)`. This triggers a focused sub-debate that loops back into the moderator for re-scoring before the main debate continues.

---

### 5. Final Scoreboard, Dramatic Events & Adversary Concession

![Final Scoreboard and Adversary Concedes](Images/_05_Img_ScoreBoard.png)

After Round 5, the complete scoreboard history is displayed - all rounds with their scores, deltas, and trend arrows. You can see the momentum swing: the adversary led consistently, but the proposer closed the gap in Round 5 (4.30 vs 4.49). The moderator decides END due to score convergence.

Two dramatic events fire: **`SCORE SHIFT!`** (a significant momentum change detected) and **`ADVERSARY CONCEDES!`** - the adversary agent itself acknowledges the proposer's architecture is production-grade: *"Your core architecture (DLS + gateway + multi-stage retrieval + citation-first generation) is production-grade. The gaps I've identified are edge-case failures that only surface under load, not fundamental design flaws. If you add adaptive timeouts, OCR quality tiers, and audit replay indexing, this ships."* Agent concession is a live dynamic graph action — when the adversary determines it cannot find further fundamental flaws, it emits a JSON `agree` action that triggers early termination.

---

### 6. Final Verdict - Architecture Brief

![Final Verdict](Images/_06_Img_Final_Verdict.png)

The **Final Verdict** is a structured architecture brief generated by the moderator. It includes:

- **Recommended Architecture** - the winning design with full rationale (OpenSearch with Document-Level Security, enforced Retrieval Gateway, multi-pass OCR with disagreement detection, evidence-span reranking)
- **Why the Alternative Was Rejected** - specific reasons the opposing proposal was dismissed, with reference to the debate evidence
- **Key Risks to Monitor** - DLS misconfiguration as a single point of failure, OCR probabilistic nature, legal "currency" of cited sources, operational burden of private cloud
- **Constraints** - query volume targets, parallelism availability, pre-computation requirements

This is the deliverable: a decision-quality architecture brief you can hand to your team, produced by three frontier models that spent multiple rounds arguing about it with real evidence.

---

### 7. Persistent Memory Update

![Memory Update](Images/_07_Img_Memory_Update.png)

After the debate ends, Archie extracts structured preferences from the session and saves them to `memory/memory.json`. In this example it captured: latency constraint (`p95 <10s`), cloud provider (`private-cloud-only`), security standards (`SOC2`), retention policy (`7 years, WORM`), data ingestion rate (`500–1,000 pages/week`), and the full technology stack discussed (Kubernetes, Argo Workflows, Kafka, Redpanda, MinIO, OpenSearch).

On the next run, these are loaded before the interview phase. The interviewer skips questions whose answers are already known, and all agents receive your established preferences as context from the first message. The merge policy is strictly additive — nothing is ever overwritten, only new unique items are appended.

---


## License

MIT
