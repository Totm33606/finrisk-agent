# FinRisk-Agent

**AI-powered credit scoring, with an AI agent you can ask questions**

A demo project showing how a classic machine learning model (predicting
whether a small business will default on a loan) can be paired with an AI
agent that explains the score in plain language, answers "what if"
questions, and justifies its recommendation — with every step logged and
checkable, not a black box.

[![CI](https://github.com/Totm33606/finrisk-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Totm33606/finrisk-agent/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Managed with uv](https://img.shields.io/badge/managed%20with-uv-de4c36)](https://docs.astral.sh/uv/)
[![LightGBM](https://img.shields.io/badge/model-LightGBM-9cbf3f)](https://lightgbm.readthedocs.io/)
[![FastMCP](https://img.shields.io/badge/protocol-FastMCP-6f42c1)](https://gofastmcp.com/)
[![LangGraph](https://img.shields.io/badge/agent-LangGraph-1c3c3c)](https://langchain-ai.github.io/langgraph/)
[![Observability: Langfuse](https://img.shields.io/badge/observability-Langfuse-orange)](https://langfuse.com/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230)](https://docs.astral.sh/ruff/)
[![Type checked: mypy --strict](https://img.shields.io/badge/mypy-strict-blue)](https://mypy-lang.org/)

https://github.com/user-attachments/assets/0191bc9e-62a0-4fcd-98ea-251818ce8eef

---

## Table of contents

- [Why this project](#why-this-project)
- [Glossary](#glossary)
- [Architecture](#architecture)
- [Repository structure](#repository-structure)
- [Quickstart](#quickstart)
  - [Option A — uv (local dev)](#option-a--uv-local-dev)
  - [Option B — Docker Compose](#option-b--docker-compose)
- [Data](#data)
- [ML pipeline](#ml-pipeline)
- [MCP server](#mcp-server)
- [Agent & observability](#agent--observability)
- [Dashboard](#dashboard)
- [Evaluation](#evaluation)
- [Testing & quality gates](#testing--quality-gates)
- [Technical choices & trade-offs](#technical-choices--trade-offs)
- [License](#license)

---

## Why this project

Most "LLM + ML" demos stop at a notebook, or wrap a model in a chatbot with
no real structure underneath. This one is built closer to how an actual
credit-decisioning tool would be:

- **The prediction itself never touches an LLM.** A classic, well-tested ML
  model (LightGBM) does the actual risk assessment — the AI only reads and
  explains its output, never invents a number.
- **The model and the agent are cleanly separated** (via MCP, see
  [Glossary](#glossary)) — you can swap out the AI agent, or retrain and
  redeploy the model, without touching the other side.
- **Every step the agent takes is logged and reviewable.** You can see
  exactly which questions it asked the model, in what order, and why it
  reached its conclusion — not a black box.
- **The dashboard shows the model's real output, not the AI's guess.** The
  charts are built from the same data the agent received, not a second,
  potentially-inconsistent call.

## Glossary

Quick definitions for terms used throughout this README — skip if you're
already familiar with them.

**Finance**
- **Default**: a company failing to repay its loan. "Credit risk" is the
  risk of this happening.
- **PD (Probability of Default)**: the model's predicted chance (0–100%)
  that a given client defaults.
- **Risk band**: a simple LOW / MEDIUM / HIGH / CRITICAL label derived from
  the PD — easier to scan than a raw percentage.
- **EBITDA margin, debt-to-equity, current ratio, DSO/DPO**: standard
  accounting ratios lenders use to gauge profitability, debt load,
  liquidity, and how fast a company pays/gets paid. Full plain-language
  definitions are in [Data](#data).

**Machine learning**
- **SHAP**: a technique that explains *why* a model gave a particular
  score, by showing how much each input pushed the prediction up or down.
- **PR-AUC / ROC-AUC**: two standard 0–1 scores (higher = better)
  summarizing how well the model tells risky and safe clients apart.
  PR-AUC is the more honest one here because actual defaults are rare —
  see [Evaluation](#evaluation).
- **Optuna / hyperparameter tuning**: an automated search that tries many
  model settings and keeps the best-performing combination, instead of
  guessing by hand.

**AI agent**
- **LLM**: a large language model (e.g. GPT) — the "brain" the agent uses
  to decide what to do next.
- **AI agent**: here, an LLM that can call tools (like "get this client's
  credit score") instead of just generating text — it looks things up
  rather than guessing.
- **MCP (Model Context Protocol)**: a standard way to expose tools (like
  the scoring model) so any AI agent can use them, independent of which
  LLM provider you pick.
- **LangGraph**: the library used to build the agent's reasoning loop
  (ask → call a tool → read the result → decide the next step, a pattern
  called *ReAct*).
- **Langfuse**: a dashboard that records everything the agent did — which
  tools it called, with what inputs/outputs, and how long each step took —
  so a human can review any past run afterwards.

## Architecture

```mermaid
flowchart TD
    UI["🖥️ React Dashboard<br/>ScoreCard · ShapChart · AgentTrace"]
    API["🤖 FastAPI Agent Layer<br/>LangGraph ReAct agent<br/>(agent/agent.py)"]
    MCP["🔌 FastMCP Server<br/>get_credit_score · get_shap_explanation<br/>simulate_financial_scenario<br/>(mcp_server/server.py)"]
    ML["📊 Core ML Pipeline<br/>preprocessing · LightGBM · SHAP<br/>(ml_pipeline/*)"]
    LF["📝 Langfuse<br/>full trace: AI calls, tool calls, cost"]

    UI -->|"REST /analyze"| API
    API -->|"MCP (stdio)"| MCP
    MCP -->|"ScoringService"| ML
    API -.->|"logged to"| LF
    MCP -.->|"logged to"| LF

    classDef ui fill:#1c3c3c,stroke:#0f2626,color:#ffffff
    classDef agent fill:#6f42c1,stroke:#4a2c85,color:#ffffff
    classDef mcp fill:#de4c36,stroke:#a6371f,color:#ffffff
    classDef ml fill:#9cbf3f,stroke:#6f8a24,color:#1a1a1a
    classDef obs fill:#f5a623,stroke:#b9750a,color:#1a1a1a

    class UI ui
    class API agent
    class MCP mcp
    class ML ml
    class LF obs
```

*(Dashboard → Agent → MCP tools → ML pipeline, with every AI call and tool
call along the way logged to Langfuse.)*

Four layers, each independently replaceable. **Why split it up this way:**

- **MCP as the seam, not framework-specific tools.** Wrapping the model
  behind MCP — instead of wiring Python functions directly into one agent
  framework — means the scoring service can be deployed and updated on its
  own schedule. Swap the agent framework for a different one later, and the
  model side doesn't change at all.
- **The agent never invents numbers.** It's instructed to ground every
  claim in a real tool call; `simulate_financial_scenario` exists so that
  "what if revenue drops 15%?" questions re-run the real model instead of
  the AI guessing a direction.
- **Explanations are computed once, on the model side, not by the AI.**
  SHAP runs inside the scoring tool, so the "why" behind a score is exact
  and reproducible no matter which AI model or agent asks for it.

## Repository structure

```
finrisk-agent/
├── pyproject.toml            # single source of truth for deps (uv-managed)
├── Makefile                  # make dataset / train / eval / mcp / api / test
├── src/
│   ├── common/schemas.py     # shared data models (ML ↔ MCP ↔ Agent ↔ UI)
│   ├── ml_pipeline/
│   │   ├── config.py         # typed, env-overridable settings
│   │   ├── make_dataset.py   # synthetic SME credit dataset generator
│   │   ├── preprocessing.py  # data cleanup/encoding pipeline
│   │   ├── train.py          # model training + hyperparameter tuning
│   │   ├── eval.py           # accuracy metrics + charts
│   │   └── shap_explainer.py # per-client "why this score" explanations
│   ├── mcp_server/
│   │   ├── scoring_service.py # model+preprocessor+explainer, testable in isolation
│   │   └── server.py          # the 3 tools + 1 resource exposed to the agent
│   └── agent/
│       ├── agent.py           # the AI agent + its web API
│       └── observability.py   # Langfuse logging setup
├── frontend/                  # Vite + React + Tailwind dashboard
│   └── src/
│       ├── App.jsx
│       └── components/{ScoreCard,ShapChart,AgentTrace}.jsx
├── tests/                     # pytest: ML pipeline, MCP server, agent (all hermetic)
├── docker/                    # Dockerfile (uv multi-stage) + compose
└── .github/workflows/ci.yml   # lint, typecheck, test on every push
```

## Quickstart

### Option A — uv (local dev)

```bash
git clone https://github.com/<you>/finrisk-agent.git
cd finrisk-agent
cp .env.example .env          # fill in your LLM + (optional) Langfuse keys

uv venv
uv pip install -e ".[dev]"
uv run python -m ml_pipeline.make_dataset   # generates data/clients.parquet (synthetic)
uv run python -m ml_pipeline.train          # trains models/lgbm_credit_risk.joblib
uv run python -m ml_pipeline.eval           # writes models/metrics.json + reports/*.png

uv run python -m mcp_server.server          # sanity-check: runs the MCP server over stdio
uv run uvicorn agent.agent:api --reload --port 8080   # starts the agent API on :8080

cd frontend && npm install && npm run dev   # dashboard on :5173
```

On Linux/macOS/WSL with `make` installed, the [Makefile](Makefile) wraps the
same `uv run ...` commands under shorter names (`make install`, `make dataset`,
`make train`, `make eval`, `make mcp`, `make api`, `make test`) — CI itself
calls `uv`/`pytest` directly (see [ci.yml](.github/workflows/ci.yml)), so
`uv` is the one command that's guaranteed to work everywhere, including
native Windows where `make` isn't available out of the box.

### Option B — Docker Compose

```bash
cp .env.example .env
make train && make eval        # bake real model artifacts before building the image
docker compose -f docker/docker-compose.yml up --build
```

This starts the agent API (`:8080`) and the dashboard (`:5173`), with model
artifacts mounted read-only into the API container.

## Data

Real bank/credit-bureau data can't ship in a public repo, so
`ml_pipeline/make_dataset.py` generates a **fake but realistic-looking**
dataset of small businesses: the numbers are drawn from random
distributions shaped to resemble real accounting figures, and whether a
company "defaults" is decided by a formula that makes financially weaker
companies default more often — while still leaving enough randomness that
it isn't a trivial lookup.

### The features (what the model sees)

One row per company (defined in `common.schemas.ClientFeatures`), 12 inputs
plus the yes/no target the model is trying to predict. The rightmost column
is implementation detail (the statistical formula used to fake the number)
— skip it if you just want the plain meaning of each field:

| Feature | Meaning | How it's faked (safe to skip) |
|---|---|---|
| `annual_revenue` | Yearly revenue, in EUR | Random, centered around a realistic SME range, 20K–50M |
| `total_debt` | Outstanding total debt, in EUR | A random 5–90% slice of that company's revenue |
| `debt_to_equity` | Debt load relative to the owners' equity — higher means more leveraged/riskier | Random, 0–12 |
| `current_ratio` | Current assets ÷ current liabilities — a liquidity check ("can they cover short-term bills?") | Random, 0.1–6 |
| `ebitda_margin` | Operating profit ÷ revenue — how profitable the core business is | Random, -40% to +45% |
| `days_payable_outstanding` | Average days the company takes to pay its own suppliers | Random, 0–180 days |
| `days_sales_outstanding` | Average days the company takes to collect payment from its customers | Random, 0–180 days |
| `late_payments_12m` | Count of late payments in the last 12 months | Random, usually 0–2 |
| `years_in_business` | Company age, in years | Random, 0.1–40 |
| `sector` | Industry (retail, manufacturing, construction, hospitality, tech, logistics) | Picked at random |
| `employees` | Headcount | Roughly scales with revenue |
| `credit_utilization` | How much of the company's approved credit line is currently used | Random, roughly 0–140% |
| `defaulted_12m` (target) | Did the company default within 12 months? | See below |

### What actually decides the target

`defaulted_12m` isn't random noise — it's driven by a weighted combination
of the signals below (each converted to a comparable "how far from average"
scale first, then run through a standard risk formula). More weight = more
influence on the outcome:

| Signal | Effect |
|---|---|
| `debt_to_equity` | ↑ risk — more debt relative to equity, riskier (strongest single driver) |
| `credit_utilization` | ↑ risk — using most of your available credit is a stress signal |
| `ebitda_margin` | ↓ risk — thinner profit margins, riskier |
| `current_ratio` | ↓ risk — less short-term liquidity, riskier |
| `late_payments_12m` | ↑ risk — a direct history-of-missed-payments signal |
| `years_in_business` | ↓ risk — younger companies default more often |
| `annual_revenue` | ↓ risk — smaller companies default more often |
| a bit of randomness | keeps the relationship realistic and imperfect, not a trivial formula |

The overall default rate is tuned to land in a realistic single-digit-to-teens
percentage. Note that **`sector`, the payment-timing fields (DPO/DSO), and
`employees` deliberately have no effect on the outcome** — they exist as
plausible-but-irrelevant context, the same way real-world data always
includes fields that don't actually matter, and (for `sector`) to exercise
the category-encoding logic in `preprocessing.py`. The exact formula lives
in `make_dataset._simulate`, for anyone who wants the full statistical
detail.

To use your own data instead: replace `make_dataset.py`'s output with a
parquet file at `data/clients.parquet` matching the schema in
`common.schemas.ClientFeatures` (plus a `defaulted_12m` target column), and
re-run `make train eval`.

## ML pipeline

```bash
uv run python -m ml_pipeline.make_dataset --n-clients 25000   # synthetic data
uv run python -m ml_pipeline.train                            # default hyperparams
uv run python -m ml_pipeline.train --tune                     # + automated tuning (40 trials)
uv run python -m ml_pipeline.eval                              # metrics + plots
```

- **Preprocessing**: cleans up missing values and encodes the `sector`
  category into numbers, fit once and reused identically for training,
  evaluation, and live scoring — so the model always sees data prepared the
  exact same way, whether it's learning or predicting.
- **Model**: LightGBM by default — a fast and widely-used model for this
  kind of spreadsheet-shaped data. It's tuned to specifically get better at
  spotting the rare "default" cases (optimizing PR-AUC — see
  [Glossary](#glossary)), with an optional automated search (Optuna) over
  its settings. A simpler, natively-interpretable alternative — logistic
  regression — is also available: set `FINRISK_MODEL_TYPE=logistic_regression`
  and re-run `train`/`eval`. Both go through the exact same pipeline,
  `ScoringService`, and dashboard — see
  [`ml_pipeline/models.py`](src/ml_pipeline/models.py) for the one place
  that decides which one gets built.
- **Explainability**: SHAP produces both a "why this score" breakdown for
  one client, and a "what matters most, overall" summary chart across all
  clients — `TreeExplainer` for LightGBM, `LinearExplainer` for logistic
  regression, picked automatically based on `FINRISK_MODEL_TYPE`.

## MCP server

Three functions the AI agent can call, plus one reference document it can
read:

```bash
uv run python -m mcp_server.server --transport stdio            # default: subprocess use
uv run python -m mcp_server.server --transport streamable-http --port 8000
```

| Tool | Purpose |
|---|---|
| `get_credit_score(client_id)` | Current default probability, risk band, and APPROVE/REVIEW/DECLINE recommendation |
| `get_shap_explanation(client_id, render_plot)` | Ranked list of what drove that score, plus an optional chart |
| `simulate_financial_scenario(client_id, ...)` | Re-scores a client under a hypothetical change (e.g. "-15% revenue") |

Plus a `finrisk://model/card` **resource** exposing the model's version,
training details, and latest accuracy metrics — background info the agent
can read once per session instead of asking for it repeatedly.

**Which clients can be scored, and why it matters.** The tools above only
serve clients from the **held-out test split** (`models/holdout_test.parquet`,
written by `make train`) — not the full synthetic dataset. This is
deliberate: the full dataset includes the rows the model was actually
trained on, and if the live demo could score those, most client_ids you'd
try would return memorized rather than genuinely out-of-sample predictions
— quietly making the demo look more accurate than the model really is. The
[Evaluation](#evaluation) numbers were always computed on this same
held-out split; this just makes sure the *live* demo has that same honesty
property.



## Agent & observability

```bash
uv run python -m agent.agent SME-000123 --question "Should we approve this client?"
uv run uvicorn agent.agent:api --reload --port 8080
```

The agent follows a loop (built with LangGraph, a pattern called *ReAct*):
read the question → decide which tool to call → read the result → repeat
until it has enough information → answer. It uses Azure OpenAI, plain
OpenAI, or a **local model** as the underlying LLM, whichever is configured
in `.env` (`_build_llm` in [`agent.py`](src/agent/agent.py) picks in that
order). No API key or cost: set `LOCAL_LLM_BASE_URL` to any
OpenAI-API-compatible local server — [Ollama](https://ollama.com)
(`http://localhost:11434/v1`), LM Studio, or llama.cpp's server all work,
as long as the model you run supports tool calling (e.g.
`qwen2.5:7b-instruct` fits comfortably in 8GB of VRAM). Every run
is logged to Langfuse with tags for the session, user, and client — so
later, anyone can look up exactly which tools were called, with what
inputs, and what the AI concluded, for any past decision. Logging is
optional: the agent still works fine locally without Langfuse configured,
it just won't be recorded.

## Dashboard

A Vite + React + Tailwind console with a deliberately non-default visual
language (ink-navy/amber "financial terminal" palette, Fraunces/JetBrains
Mono type pairing) built around three panels: the score gauge
(`ScoreCard`), the "why this score" chart (`ShapChart`), and a live feed of
the agent's tool calls as they happen (`AgentTrace`) — so you can watch the
AI's reasoning step by step instead of just waiting for a final answer.

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173, expects the agent API on :8080
```

## Evaluation

Numbers below are from an actual run of `make dataset train eval` on this
repository (25,000 fake companies, 13.7% of them "defaulted", default model
settings, no extra tuning):

| Metric | Value |
|---|---|
| ROC-AUC | **0.917** |
| PR-AUC (average precision) | **0.700** |
| Precision @ threshold=0.30 | 0.432 |
| Recall @ threshold=0.30 | 0.873 |
| F1 @ threshold=0.30 | 0.578 |
| Base rate (test set) | 13.7% |

**In plain terms:** given a random risky company and a random safe one, the
model ranks them correctly about 92% of the time (ROC-AUC). Specifically at
hunting down the rare risky ones, it beats random guessing by roughly 5×
(PR-AUC). At the current decision threshold, it flags 87% of the companies
that actually go on to default (recall) — at the cost of also flagging some
that would have been fine (only 43% of flagged companies actually default,
i.e. precision). That's a deliberately cautious setting: missing a real
default is more costly than double-checking a client who turns out fine.
You can shift that trade-off with the `FINRISK_DECISION_THRESHOLD` setting.

*Why PR-AUC and not just ROC-AUC?* When the thing you're predicting is rare
— like a 13.7% default rate — ROC-AUC can look artificially good, while
PR-AUC stays honest under that imbalance. See
`ml_pipeline/train.py::_cv_average_precision` for the full statistical
rationale, if you want it. Full charts (ROC curve, PR curve, confusion
matrix, and SHAP summary plots) are written to `reports/` by `make eval`.

Running the tuning step (`make train --tune`, ~10-15 minutes) usually
improves these numbers further; exact gains depend on the random seed and
aren't hard-coded here, to keep this table honest about what the *default*
settings produce.

## Testing & quality gates

**Backend (Python):**

```bash
uv run pytest         # hermetic (trains a tiny model in-fixture, no disk artifacts required)
uv run ruff check src tests
uv run mypy src
```

(`make test` / `make lint` / `make typecheck` are equivalent shortcuts, where `make` is available.)

**Frontend (`frontend/`):** linting and formatting are two separate, deliberately
non-overlapping tools — [ESLint](https://eslint.org/) for code-quality rules
(unused variables, React hooks misuse, ...), [Biome](https://biomejs.dev/)
for formatting only (its own linter is switched off in
[`frontend/biome.json`](frontend/biome.json), to avoid two tools disagreeing
about the same rule). Biome was picked over Prettier specifically to mirror
the Python side's own tooling philosophy — a single fast Rust binary instead
of a Node-based formatter, the same reasoning behind choosing `uv`/`ruff`
(see [Technical choices](#technical-choices--trade-offs)).

```bash
cd frontend
npm run lint           # eslint — code-quality rules
npm run format:check   # biome — formatting only, no autofix
npm run format          # biome --write — autofix formatting
npm run build           # production build, also catches real compile errors
```

**Neither of these ran automatically until a `frontend-quality` CI job was
added** — before that, `npm run lint`/`format` only ever ran if someone
remembered to type the command by hand. Now `.github/workflows/ci.yml` runs
`npm run lint`, `npm run format:check`, and `npm run build` on every
push/PR, exactly the same way the Python job runs `ruff`/`mypy`/`pytest` —
so a broken frontend fails CI the same way a broken backend does, instead of
silently shipping.

This repository's own `src/` passes `ruff check`, `ruff format --check`,
and `mypy --strict` with zero errors, and no dependency is left at a
"latest" floor that isn't actually validated — see the trade-offs section
below.

## Technical choices & trade-offs

### Stack justification

Each major dependency was picked over a real alternative, not by default:

| Layer | Choice | Why |
|---|---|---|
| Tabular model | LightGBM (default), logistic regression (configurable) | LightGBM is fast and well-tested for spreadsheet-shaped data like this. Logistic regression is available via `FINRISK_MODEL_TYPE` for when native interpretability (a direct coefficient per feature) matters more than the last bit of accuracy — see [`ml_pipeline/models.py`](src/ml_pipeline/models.py). |
| Hyperparameter search | Optuna (`TPESampler`) | Smarter than a blind grid search — it learns from earlier attempts to try more promising settings next, and respects a trial/time budget. Chosen over Hyperopt/Ray Tune for being lightweight and dependency-free (no cluster needed). |
| Explainability | SHAP (`TreeExplainer`) | The standard, exact, and fast way to explain individual predictions from tree-based models like LightGBM — versus the slower, sampling-based alternative (Kernel SHAP) meant for models where an exact method isn't available. |
| Model↔agent boundary | MCP (via FastMCP) | Keeps the scoring model fully independent of whichever AI agent or provider uses it — swap the agent framework later, and the model side is untouched. The alternative (wiring functions straight into one agent framework) would lock the two together. |
| Agent orchestration | LangGraph | A ready-made, tested "ask → call a tool → read result → repeat" loop, instead of hand-writing that control flow (more code, more places for bugs). |
| Observability | Langfuse | Purpose-built for recording what an AI agent did — tool calls, inputs/outputs, cost — which is exactly the audit trail a finance use case needs. It's optional: the agent still runs without it. |
| Configuration | pydantic-settings | One typed, validated settings file shared by every script, so they can't silently drift out of sync with each other. |
| CLIs | typer | Turns ordinary Python functions into command-line tools with almost no extra code. |
| Packaging & running | uv | One fast tool for creating environments, installing dependencies, and running scripts — replacing the usual pip + venv combo, and working the same way on Windows, macOS, and Linux. |
| Frontend formatting | Biome | A single fast Rust binary instead of Prettier — deliberately mirrors the `uv`/`ruff` reasoning above, so the tooling philosophy is consistent on both sides of the stack. Scoped to formatting only (its linter is off) so it doesn't duplicate/conflict with ESLint's code-quality rules. |

### Design decisions & trade-offs

- **"What if" questions always re-run the real model — never an AI guess.**
  A hypothetical question (e.g. "what if revenue drops 15%?") is answered
  by tweaking that client's numbers and asking the real model again — the
  answer the user sees is always a genuine model output, never the AI's
  estimate of what the model would probably say.
- **Pinned, tested dependency versions.** A few AI-related packages
  (`langfuse`, `langchain`, `langgraph-prebuilt`) are pinned to specific
  version ranges rather than "always latest", because their newest releases
  don't currently work together — pinning avoids a future install silently
  breaking. The exact combination in `pyproject.toml` has been run
  end-to-end against the full test suite.
- **Synthetic data by necessity, not preference.** The data generator is
  clearly documented as a stand-in — see [Data](#data) for how to plug in
  real data. The accuracy numbers in this README are honestly
  synthetic-data numbers, not a claim about real-world performance.
- **Local subprocess connection (stdio) by default**, rather than exposing
  the scoring server over HTTP. Simpler and more contained for a tool
  that's spawned locally by the agent; an HTTP option
  (`streamable-http`) is available for remote or multi-consumer setups.

## License

MIT — see [LICENSE](LICENSE).
