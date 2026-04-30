import os
import json
import re
import time
from pathlib import Path
from typing import NamedTuple, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, InvalidArgument, Unauthenticated

# Load .env from the script's directory
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

app = FastAPI(title="PROMETHEUS - Prompt Evolution Engine")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.getenv("GEMINI_API_KEY", "").strip().strip('"').strip("'")
if not api_key:
    raise RuntimeError(
        f"GEMINI_API_KEY not found. Ensure .env file at {env_path} contains: GEMINI_API_KEY=..."
    )

# Initialize Gemini API
genai.configure(api_key=api_key)

# Auto-detect available model
try:
    available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
    if available_models:
        model_name = available_models[0]
    else:
        model_name = 'gemini-1.5-flash'
except Exception:
    model_name = 'gemini-1.5-flash'

model = genai.GenerativeModel(model_name)


class Intent(NamedTuple):
    surface: str
    deep: str
    implicit: list
    antigoals: list


class Memory:
    def __init__(self):
        self.wins = []
        self.losses = []
        self.dead_zones = set()

    def log_loss(self, strategy: str, reason: str):
        self.losses.append((strategy, reason))
        self.dead_zones.add(f"{strategy}:{reason[:40]}")

    def get_bias(self) -> list:
        return [z.split(":")[0] for z in self.dead_zones if z.split(":")[0] in ["EXPAND", "TIGHTEN", "ROLE_INJECT", "PATCH", "ENTROPY"]]


class PrometheusRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Raw user prompt to optimize")
    max_iterations: int = Field(default=3, ge=1, le=5)
    theta: float = Field(default=0.85, ge=0.5, le=1.0)


class PrometheusResponse(BaseModel):
    optimized_prompt: str
    why_better: str
    intent: dict
    mutations: list
    tournament_results: list
    reflection_scores: list
    iterations: int
    trace: list


def call_openai(system_prompt: str, user_message: str, max_retries: int = 5) -> str:
    """Call Google Gemini API with aggressive exponential backoff retry logic."""
    for attempt in range(max_retries):
        try:
            full_prompt = f"{system_prompt}\n\nUser: {user_message}"
            response = model.generate_content(full_prompt)
            return response.text.strip()
        except ResourceExhausted:
            if attempt < max_retries - 1:
                # Aggressive backoff: 3s, 6s, 12s, 24s, 48s
                wait_time = 3 * (2 ** attempt)
                time.sleep(wait_time)
                continue
            raise HTTPException(status_code=429, detail="Gemini API rate limit exceeded. Free tier limit reached. Please wait a few minutes and try again.")
        except Unauthenticated:
            raise HTTPException(status_code=401, detail="Invalid Gemini API key. Check GEMINI_API_KEY.")
        except InvalidArgument as e:
            raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")
    
    raise HTTPException(status_code=500, detail="Failed to get response from Gemini after retries.")


def extract_json_from_response(response_text: str) -> dict:
    """Extract JSON from response, handling markdown code blocks."""
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code blocks
        import re
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON object in response
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}')
        if start_idx != -1 and end_idx != -1:
            try:
                return json.loads(response_text[start_idx:end_idx + 1])
            except json.JSONDecodeError:
                pass
        
        raise ValueError(f"Could not extract JSON from response: {response_text[:200]}")


def extract_intent(prompt: str) -> Intent:
    """Extract latent intent: surface, deep, implicit, antigoals."""
    system = """You are an intent decomposition engine. Analyze the user's prompt and return ONLY valid JSON with:
    - surface: what was literally asked
    - deep: what success actually looks like for the user
    - implicit: list of unstated assumptions or constraints
    - antigoals: list of things a bad response would do
    
    Return ONLY JSON, no markdown, no code blocks, no explanation."""
    
    response = call_openai(system, f"User prompt: {prompt}")
    
    try:
        data = extract_json_from_response(response)
        return Intent(
            surface=data.get("surface", ""),
            deep=data.get("deep", ""),
            implicit=data.get("implicit", []),
            antigoals=data.get("antigoals", []),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse intent extraction response: {str(e)}")


def mutate(prompt: str, intent: Intent, memory: Memory) -> dict:
    """Generate N candidate variants using 3 core mutation strategies."""
    avoid = memory.get_bias()
    strategies = {
        "EXPAND": f"""Rewrite this prompt with added context, domain framing, and background knowledge that would unlock a richer response.
            Original: {prompt}
            Deep intent: {intent.deep}""",
        
        "TIGHTEN": f"""Rewrite this prompt with explicit output constraints (format, length, structure, tone).
            Remove ambiguity and add measurable success criteria.
            Original: {prompt}""",
        
        "ROLE_INJECT": f"""Rewrite this prompt by prepending an expert persona aligned to the deep intent.
            Deep intent: {intent.deep}
            Original: {prompt}""",
    }
    
    variants = {}
    for name, instruction in strategies.items():
        if name in avoid:
            continue
        system = "You are an expert prompt engineer. Return only the rewritten prompt, nothing else."
        variants[name] = call_openai(system, instruction)
    
    return variants


def score_prompt(prompt: str, intent: Intent) -> dict:
    """Score prompt against intent across 4 axes."""
    system = f"""Score this prompt's quality against the stated intent. Return ONLY JSON with:
    intent_align: closeness to deep intent (0-1)
    specificity: level of detail and constraints (0-1)
    robustness: resistance to misinterpretation (0-1)
    clarity: how quickly the LLM understands it (0-1)
    
    Deep intent: {intent.deep}
    Implicit constraints: {', '.join(intent.implicit) or 'none'}
    Antigoals: {', '.join(intent.antigoals) or 'none'}
    
    Return ONLY JSON, no markdown, no code blocks."""
    
    response = call_openai(system, f"Prompt to score: {prompt}")
    
    try:
        return extract_json_from_response(response)
    except Exception:
        return {
            "intent_align": 0.7,
            "specificity": 0.7,
            "robustness": 0.7,
            "clarity": 0.7,
        }


def tournament(candidates: dict, intent: Intent, memory: Memory, trace: list) -> tuple:
    """1v1 tournament with elimination."""
    pool = list(candidates.items())
    round_results = []
    
    while len(pool) > 1:
        (name_a, prompt_a) = pool.pop()
        (name_b, prompt_b) = pool.pop()
        
        scores_a = score_prompt(prompt_a, intent)
        scores_b = score_prompt(prompt_b, intent)
        
        total_a = sum(scores_a.values())
        total_b = sum(scores_b.values())
        
        if total_a >= total_b:
            winner, winner_name, loser_name, loser_score = prompt_a, name_a, name_b, scores_b
        else:
            winner, winner_name, loser_name, loser_score = prompt_b, name_b, name_a, scores_a
        
        weakest = min(loser_score, key=loser_score.get)
        memory.log_loss(loser_name, weakest)
        
        round_results.append({
            "matchup": f"{name_a} vs {name_b}",
            "winner": winner_name,
            "winner_scores": (scores_a.copy() if winner_name == name_a else scores_b.copy()),
            "loser_scores": (scores_b.copy() if winner_name == name_a else scores_a.copy()),
        })
        
        pool.insert(0, (winner_name, winner))
    
    trace.append({"round": "tournament", "results": round_results})
    return pool[0][1], round_results


def reflect(prompt: str, intent: Intent) -> dict:
    """Self-critique pass: identify gaps and score fit."""
    system = f"""Critique this prompt against its intended goal. Return ONLY JSON with:
    overall_score: float (0-1) aggregate fitness
    intent_align: how well it captures the deep intent
    specificity: how clear the success criteria are
    robustness: how well it avoids the antigoals
    clarity: how easily understood by an LLM
    gaps: list of specific weaknesses
    patches: list of exact improvements to make
    explanation: why this is better than the original
    
    Deep intent: {intent.deep}
    Implicit constraints: {', '.join(intent.implicit) or 'none'}
    Antigoals: {', '.join(intent.antigoals) or 'none'}
    
    Return ONLY JSON, no markdown, no code blocks."""
    
    response = call_openai(system, f"Prompt to critique: {prompt}")
    
    try:
        return extract_json_from_response(response)
    except Exception:
        return {
            "overall_score": 0.8,
            "intent_align": 0.8,
            "specificity": 0.8,
            "robustness": 0.8,
            "clarity": 0.8,
            "gaps": [],
            "patches": [],
            "explanation": "Reflection failed to parse; accepting current prompt.",
        }


def apply_patches(prompt: str, patches: list) -> str:
    """Apply critique patches to refine prompt."""
    if not patches:
        return prompt
    
    system = "You are a prompt refinement specialist. Apply the given patches carefully while preserving the prompt's core intent. Return only the refined prompt."
    instruction = f"""Apply these refinements to the prompt:
    Patches to apply: {', '.join(patches)}
    Original prompt: {prompt}"""
    
    return call_openai(system, instruction)


@app.post("/optimize", response_model=PrometheusResponse)
async def prometheus_optimize(request: PrometheusRequest):
    """Execute PROMETHEUS prompt evolution algorithm."""
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
    
    memory = Memory()
    trace = []
    prompt_0 = request.prompt.strip()
    current_best = prompt_0
    
    # STEP 1: Extract intent
    intent = extract_intent(prompt_0)
    trace.append({
        "iteration": 0,
        "step": "intent_extraction",
        "intent": intent._asdict(),
    })
    
    tournament_results_all = []
    reflection_scores_all = []
    
    # STEP 2-4: Evolutionary loop
    for iteration in range(request.max_iterations):
        # Mutate
        candidates = mutate(current_best, intent, memory)
        trace.append({
            "iteration": iteration + 1,
            "step": "mutation",
            "strategies_applied": list(candidates.keys()),
        })
        
        # Tournament
        champion, tournament_results = tournament(candidates, intent, memory, trace)
        tournament_results_all.append(tournament_results)
        
        # Reflect
        critique = reflect(champion, intent)
        reflection_scores_all.append({
            "iteration": iteration + 1,
            "scores": {
                "overall": critique.get("overall_score", 0),
                "intent_align": critique.get("intent_align", 0),
                "specificity": critique.get("specificity", 0),
                "robustness": critique.get("robustness", 0),
                "clarity": critique.get("clarity", 0),
            },
        })
        
        trace.append({
            "iteration": iteration + 1,
            "step": "reflection",
            "score": critique.get("overall_score", 0),
            "gaps": critique.get("gaps", []),
        })
        
        # Check if fitness threshold met
        if critique.get("overall_score", 0) >= request.theta:
            return PrometheusResponse(
                optimized_prompt=champion,
                why_better=critique.get("explanation", "Optimized prompt achieved target fitness."),
                intent=intent._asdict(),
                mutations=[{"iteration": i + 1, "strategies": list(candidates.keys())} for i, _ in enumerate(range(iteration + 1))],
                tournament_results=tournament_results_all,
                reflection_scores=reflection_scores_all,
                iterations=iteration + 1,
                trace=trace,
            )
        
        # Patch and continue
        current_best = apply_patches(champion, critique.get("patches", []))
        memory.wins.append((champion, critique.get("overall_score", 0)))
    
    # Max iterations reached
    return PrometheusResponse(
        optimized_prompt=current_best,
        why_better="Max iterations reached. Best available output after evolutionary process.",
        intent=intent._asdict(),
        mutations=[{"iteration": i + 1, "strategies": list(memory.get_bias())} for i in range(request.max_iterations)],
        tournament_results=tournament_results_all,
        reflection_scores=reflection_scores_all,
        iterations=request.max_iterations,
        trace=trace,
    )

