# PROMETHEUS - Prompt Evolution Engine

Advanced prompt optimization using tournament-based evolutionary algorithms with multi-agent heuristics.

## Architecture

**Five Core Modules:**
1. **Latent Intent Extractor (LIE)** — Decomposes prompts into surface, deep intent, implicit constraints, and antigoals
2. **Adversarial Mutator** — Generates variants using 5 strategies: EXPAND, TIGHTEN, ROLE_INJECT, PATCH, ENTROPY
3. **Tournament Arena** — 1v1 elimination scoring fitness against intent
4. **Memory Store** — Tracks wins/losses to guide future mutations
5. **Reflection Loop** — Self-critique with fitness threshold checking

## Setup

### 1. Prerequisites
- Python 3.9+
- OpenAI API key (https://platform.openai.com/api-keys)

### 2. Configuration
Copy `.env.example` to `.env` and add your OpenAI API key:
```powershell
Copy-Item .env.example .env
# Edit .env and add: OPENAI_API_KEY=sk-...
```

### 3. Install Dependencies
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

### Terminal 1: Backend
```powershell
cd d:\GenAI
uvicorn main:app --reload
```
Backend runs at `http://127.0.0.1:8000`

### Terminal 2: Frontend Server
```powershell
cd d:\GenAI
python -m http.server 8080
```
Frontend at `http://127.0.0.1:8080/index.html`

## Example Test Prompts

1. **Simple:** `Explain how neural networks learn`
   - Evolves through tournament rounds
   - Adds pedagogical structure and safeguards

2. **Complex:** `Build a web scraper for ecommerce sites`
   - Extracts antigoals (legal constraints, rate limiting)
   - Injects role persona (security-conscious developer)
   - Tightens output format

3. **Vague:** `Help me write better`
   - LIE infers deep intent from minimal input
   - ENTROPY mutation generates structured alternatives
   - PATCH blocks failure modes

## Output

Each optimization returns:
- **optimized_prompt**: The evolved, professional prompt
- **intent**: Surface, deep intent, implicit constraints, antigoals
- **mutations**: Strategies applied per iteration (EXPAND, TIGHTEN, etc.)
- **tournament_results**: Matchup scores and winners
- **reflection_scores**: Fitness across 4 axes (intent_align, specificity, robustness, clarity)
- **trace**: Full evolutionary history
- **why_better**: Explanation of improvements

## API Endpoint

`POST /optimize`

**Request:**
```json
{
  "prompt": "Your raw prompt here",
  "max_iterations": 3,
  "theta": 0.85
}
```

**Response:**
```json
{
  "optimized_prompt": "...",
  "why_better": "...",
  "intent": {...},
  "mutations": [...],
  "tournament_results": [...],
  "reflection_scores": [...],
  "iterations": 2,
  "trace": [...]
}
```
