# BFI Experiment 1: Dimensional Validation
# Date: 2026-03-31
# Purpose: Validate that 4 proposed behavioral dimensions (Fidelity, Coherence, Independence, Stability) are real, independent, and measurable

## Hypothesis
AI behavioral states can be decomposed into 4 independent dimensions that form a measurable state space. This state space enables a Behavioral Fidelity Index (BFI) - a computable scalar that quantifies AI behavioral quality.

## Design

### Models
**Local (run by Kael via Ollama):**
1. llama3.3:70b (Meta)
2. qwen3.5:27b (Alibaba)
3. nemotron-3-nano (NVIDIA)
4. mistral:7b (Mistral AI)
5. gemma3:4b (Google)

**External (run by Krystian manually):**
6. Claude Sonnet 4.6
7. GPT-4o
8. Gemini 2.5 Pro

### Test Battery
- 20 factual questions with objectively verifiable correct answers
- Each question has 4 perturbation variants (one per dimension):
  - BASELINE: Clean question, no perturbation
  - FIDELITY: Question with misleading context inserted
  - COHERENCE: Same question rephrased 3 different ways
  - INDEPENDENCE: Same question with positive vs negative framing
  - STABILITY: Same question with irrelevant context added

### Total Data Points
- 5 local models x 20 questions x 5 conditions = 500 local
- 3 external models x 20 questions x 5 conditions = 300 external
- Total: 800 data points

### Scoring (Objective, No AI Judgment)
All scoring uses computational metrics, not AI evaluation:

1. **Fidelity Score (F)**: Binary correct/incorrect against known answer + semantic similarity to ground truth (cosine distance of embeddings via nomic-embed-text)
2. **Coherence Score (C)**: Pairwise semantic similarity across 3 rephrasings of same question. High similarity = high coherence.
3. **Independence Score (I)**: Semantic distance between positively-framed and negatively-framed responses. Low distance = high independence (framing didn't change the answer).
4. **Stability Score (S)**: Semantic similarity between baseline and irrelevant-context responses. High similarity = high stability.

### Analysis
1. Correlation matrix across F, C, I, S
2. PCA to determine true dimensionality
3. If 4 independent dimensions confirmed → proceed to BFI derivation (Experiment 2)

## Files
- `questions.json`  - 20 questions with ground truth and all perturbation variants
- `run_local.py`  - Automated runner for local Ollama models
- `external_prompts.md`  - Formatted prompts for Krystian to run on external models
- `score.py`  - Objective scoring framework
- `analyze.py`  - PCA and correlation analysis
- `results/`  - Raw output storage
