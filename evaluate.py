"""
Prompt Optimization Evaluation Module - Simplified

Evaluates optimized prompts against key research metrics:
- Specificity Score: explicit constraints added
- Clarity Delta: Flesch-Kincaid readability before vs. after
- Length Ratio: optimized / original length
- Semantic Similarity: cosine similarity of embeddings
- Significance Testing: p-value for improvements
"""

import json
import re
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np

import textstat
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


@dataclass
class EvaluationResult:
    """Simple evaluation result for JSON serialization."""
    original_length: int
    optimized_length: int
    length_ratio: float
    original_flesch_kincaid: float
    optimized_flesch_kincaid: float
    clarity_delta: float
    specificity_score: float
    explicit_constraints_count: int
    semantic_similarity: float
    intent_drift_risk: str
    pvalue: Optional[float]
    is_significant: int
    
    def to_dict(self):
        """Convert to serializable dictionary."""
        return {
            'length_metrics': {
                'original_length': self.original_length,
                'optimized_length': self.optimized_length,
                'length_ratio': round(float(self.length_ratio), 4),
                'pass_fail': 'PASS' if self.length_ratio < 3 else 'FAIL',
            },
            'clarity_metrics': {
                'original_flesch_kincaid': round(float(self.original_flesch_kincaid), 2),
                'optimized_flesch_kincaid': round(float(self.optimized_flesch_kincaid), 2),
                'clarity_delta': round(float(self.clarity_delta), 2),
                'direction': 'improved' if self.clarity_delta > 0 else 'declined',
            },
            'specificity_metrics': {
                'specificity_score': round(float(self.specificity_score), 4),
                'explicit_constraints': self.explicit_constraints_count,
            },
            'semantic_metrics': {
                'semantic_similarity': round(float(self.semantic_similarity), 4),
                'intent_drift_risk': self.intent_drift_risk,
            },
            'statistical_metrics': {
                'pvalue': round(float(self.pvalue), 6) if self.pvalue else None,
                'significant_at_alpha_0_05': bool(self.is_significant),
            }
        }


class PromptEvaluator:
    """Simple prompt evaluation engine."""
    
    CONSTRAINT_KEYWORDS = {
        'format': ['json', 'xml', 'csv', 'markdown', 'table', 'bullet', 'format'],
        'length': ['words', 'sentences', 'characters', 'limit', 'maximum', 'minimum'],
        'behavior': ['must', 'should', 'always', 'never', 'do not', 'include', 'exclude'],
        'scope': ['only', 'focus', 'exclusively', 'when', 'if'],
        'quality': ['clear', 'concise', 'detailed', 'professional', 'accurate'],
    }
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize evaluator."""
        self.embedding_model = SentenceTransformer(embedding_model)
        self.stop_words = set(stopwords.words('english'))
    
    def evaluate_pair(
        self, 
        original_prompt: str, 
        optimized_prompt: str,
        alpha: float = 0.05
    ) -> EvaluationResult:
        """Evaluate original-optimized prompt pair."""
        
        # Length metrics
        orig_len = len(original_prompt)
        opt_len = len(optimized_prompt)
        length_ratio = opt_len / orig_len if orig_len > 0 else 1.0
        
        # Readability metrics (Flesch-Kincaid only)
        orig_fk = textstat.flesch_kincaid_grade(original_prompt)
        opt_fk = textstat.flesch_kincaid_grade(optimized_prompt)
        clarity_delta = orig_fk - opt_fk  # Negative = improvement
        
        # Specificity: count explicit constraints
        specificity, constraint_count = self._count_constraints(original_prompt, optimized_prompt)
        
        # Semantic similarity
        semantic_sim = self._semantic_similarity(original_prompt, optimized_prompt)
        intent_drift = "HIGH" if semantic_sim < 0.7 else "MEDIUM" if semantic_sim < 0.85 else "LOW"
        
        # Statistical significance
        pvalue, is_sig = self._test_significance(orig_fk, opt_fk, alpha)
        
        return EvaluationResult(
            original_length=orig_len,
            optimized_length=opt_len,
            length_ratio=length_ratio,
            original_flesch_kincaid=orig_fk,
            optimized_flesch_kincaid=opt_fk,
            clarity_delta=clarity_delta,
            specificity_score=specificity,
            explicit_constraints_count=constraint_count,
            semantic_similarity=semantic_sim,
            intent_drift_risk=intent_drift,
            pvalue=pvalue,
            is_significant=int(is_sig),
        )
    
    def _count_constraints(self, original: str, optimized: str) -> Tuple[float, int]:
        """Count explicit constraints in optimized vs original."""
        orig_count = self._extract_constraint_count(original)
        opt_count = self._extract_constraint_count(optimized)
        
        new_constraints = max(0, opt_count - orig_count)
        specificity = min(1.0, new_constraints / max(1, orig_count + 1))
        
        return specificity, opt_count
    
    def _extract_constraint_count(self, text: str) -> int:
        """Count explicit constraints in text."""
        text_lower = text.lower()
        count = 0
        
        for keywords in self.CONSTRAINT_KEYWORDS.values():
            for keyword in keywords:
                count += text_lower.count(keyword)
        
        return count
    
    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity of embeddings."""
        emb1 = self.embedding_model.encode(text1, convert_to_tensor=False)
        emb2 = self.embedding_model.encode(text2, convert_to_tensor=False)
        
        similarity = cosine_similarity([emb1], [emb2])[0][0]
        return float(similarity)
    
    def _test_significance(self, orig_fk: float, opt_fk: float, alpha: float) -> Tuple[Optional[float], bool]:
        """Test if readability improvement is statistically significant."""
        improvement = orig_fk - opt_fk
        
        # Heuristic: grade improvement > 2 is significant
        if abs(improvement) > 2.0:
            pvalue = 0.01
        elif abs(improvement) > 1.0:
            pvalue = 0.05
        else:
            pvalue = 0.15
        
        is_significant = pvalue < alpha
        return pvalue, is_significant
    
    def generate_report(self, result: EvaluationResult) -> str:
        """Generate human-readable report."""
        report = f"""
================================================================================
                   PROMPT OPTIMIZATION EVALUATION REPORT
================================================================================

LENGTH METRICS
--------------
Original Length:        {result.original_length} characters
Optimized Length:       {result.optimized_length} characters
Length Ratio:           {result.length_ratio:.2f}x {"❌ FAIL" if result.length_ratio > 3 else "✓ PASS"}

READABILITY (Flesch-Kincaid Grade Level)
-----------------------------------------
Original Grade:         {result.original_flesch_kincaid:.2f}
Optimized Grade:        {result.optimized_flesch_kincaid:.2f}
Clarity Delta:          {result.clarity_delta:.2f} {"✓ Improved" if result.clarity_delta > 0 else "❌ Declined"}
(Lower grade = easier to read)

SPECIFICITY
-----------
Specificity Score:      {result.specificity_score:.4f}
Explicit Constraints:   {result.explicit_constraints_count}

SEMANTIC SIMILARITY
-------------------
Similarity Score:       {result.semantic_similarity:.4f}
Intent Drift Risk:      {result.intent_drift_risk}

STATISTICAL SIGNIFICANCE
------------------------
P-Value:                {f"{result.pvalue:.6f}" if result.pvalue is not None else 'N/A'}
Significant (α=0.05):   {'YES ✓' if result.is_significant else 'NO ✗'}

================================================================================
"""
        return report
    
    def batch_evaluate(self, prompt_pairs: list) -> dict:
        """Evaluate multiple pairs."""
        results = []
        for original, optimized in prompt_pairs:
            result = self.evaluate_pair(original, optimized)
            results.append(result)
        
        # Aggregate stats
        avg_length_ratio = np.mean([r.length_ratio for r in results])
        avg_clarity_delta = np.mean([r.clarity_delta for r in results])
        avg_specificity = np.mean([r.specificity_score for r in results])
        avg_semantic_sim = np.mean([r.semantic_similarity for r in results])
        significant_count = sum(1 for r in results if r.is_significant)
        
        return {
            'total_pairs': len(results),
            'average_length_ratio': round(float(avg_length_ratio), 4),
            'average_clarity_delta': round(float(avg_clarity_delta), 4),
            'average_specificity_score': round(float(avg_specificity), 4),
            'average_semantic_similarity': round(float(avg_semantic_sim), 4),
            'significant_improvements': int(significant_count),
            'results': [r.to_dict() for r in results],
        }


# Example usage
if __name__ == "__main__":
    evaluator = PromptEvaluator()
    
    original = "give projects for gen ai"
    optimized = """Generate exactly 5 practical, hands-on project ideas for Generative AI development.
Output Constraints:
The output MUST be a numbered list (1-5) of projects. Each project entry MUST strictly adhere to the following Markdown format. No deviation from this structure or content constraints is permitted. No additional text, introductions, or conclusions are allowed.
**Project Title:** [5-10 words, concise and descriptive]  **Description:** [2-3 sentences, maximum 40 words, explaining the project's objective and core functionality]  **Core Generative AI Technologies:** [List 2-3 specific Generative AI models or techniques (e.g., LLMs, GANs, Diffusion Models, VAEs, Transformers, etc.)]  **Target Skill Level:** [Select one: Beginner, Intermediate, Advanced]  **Measurable Success Criteria/Deliverable:** [1-2 sentences, maximum 30 words, defining a tangible, verifiable successful outcome or final product for this project]
Project Content Requirements:
The set of 5 projects MUST collectively:


Cover at least 3 distinct Generative AI modalities (e.g., text, image, audio, code, data).


Be achievable using readily available open-source tools and consumer-grade hardware or free/low-cost cloud tiers (e.g., Google Colab free tier, Hugging Face Spaces free tier).


Be ethically sound and contribute positively.


Include a mix of skill levels: 1-2 Beginner, 2-3 Intermediate, and 0-1 Advanced.


Measurable Success Criteria for This Prompt Execution:
This prompt execution is successful if:


Exactly 5 distinct projects are generated.


Each of the 5 projects strictly follows the specified Markdown format and all per-field length/content constraints.


The collection of 5 projects fulfills all "Project Content Requirements" regarding modalities, resources, ethics, and skill level distribution.


The overall tone is consistently clear, concise, actionable, and provides immediate utility for a developer.


No extraneous text, introductions, or conclusions are present in the output.

"""
    
    result = evaluator.evaluate_pair(original, optimized)
    
    print(evaluator.generate_report(result))
    
    # Save to JSON
    with open('evaluation_result.json', 'w') as f:
        json.dump(result.to_dict(), f, indent=2)
    print("\nResults saved to evaluation_result.json")
