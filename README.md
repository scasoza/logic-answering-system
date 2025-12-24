# Logic Answering System

A reasoning system that uses large language models to approach truth through iterative disambiguation, reasoning, and verification.

## Philosophy

This system is based on the perspective that:

- **Reality exists** and we access it through data
- **Our model of reality** is what we navigate in
- **Questions are pointers** - "I think there is something shaped like this and I want to find it"
- **Truth is also a pointer** - it describes a shape that must resolve to a single target
- **Truth cannot be ambiguous** - if a truth claim points to multiple things, it is undefined

## Architecture

### Ambiguity Lowering Step

The system first refines questions to reduce ambiguity:

1. **Determine Ambiguity** - Analyze if the question contains terms that could point to multiple things
2. **Break Apart Components** - Decompose the sentence into semantic units
3. **Specificity Queries** - Analyze each component for multiple meanings (parallel)
4. **Response to Ambiguous Terms** - Resolve ambiguities or ask user for clarification
5. **Cohesive Updated Question** - Produce a refined, unambiguous question

Steps 2-4 repeat until the question is unambiguous.

### Reasoning Step

Uses adversarial back-and-forth to approach truth:

1. **Initial Reasoning** (Opus) - Generate reasoning to find the answer
2. **Claim Extraction** - Break reasoning into atomic, verifiable claims
3. **Claim Verification** - Verify each claim with recursive verification
   - Verdicts: TRUE, FALSE, UNDEFINED
   - UNDEFINED categories: AMBIGUOUS, EMPIRICALLY_UNKNOWN, AXIOM_DEPENDENT, LOGICALLY_UNDECIDABLE
4. **Completion Check** (Opus) - Determine if the reasoning answers the question

### Termination Conditions

1. Reasoning chain verified TRUE and answers the question
2. Reasoning chain proves the question cannot be answered (verified TRUE)
3. Budget/depth limit reached

## Model Assignments

| Role | Model |
|------|-------|
| Initial reasoning | Claude Opus 4.5 |
| Completion check | Claude Opus 4.5 |
| All other calls | Gemini 3 Flash |

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd logic-answering-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core dependencies
pip install -r requirements.txt

# Optional: Install audio transcription (requires ~1GB+ disk space)
pip install -r requirements-audio.txt

# Copy environment file and add API keys
cp .env.example .env
# Edit .env with your API keys
```

## Usage

### Web Interface

```bash
python main.py serve
# Open http://localhost:8000 in your browser
```

### CLI - Single Question

```bash
python main.py ask "What is the meaning of life?"
python main.py ask "Is P=NP?" -v  # Verbose output
```

### CLI - Interactive Mode

```bash
python main.py interactive
```

## Configuration

Environment variables in `.env`:

```
ANTHROPIC_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here

# Optional
SOFT_BUDGET_LIMIT=1.00      # Warning threshold (USD)
HARD_BUDGET_LIMIT=5.00      # Hard stop threshold (USD)
MAX_VERIFICATION_DEPTH=5    # Max recursive verification depth
MAX_DISAMBIGUATION_ITERATIONS=10
```

## Cost Tracking

All API calls are logged with:
- Model used
- Token counts
- Computed cost
- Timestamp
- Call type

Logs are saved as JSON in the `logs/` directory.

## UI Features

- **Collapsible tree view** of the entire reasoning process
- **Color coding**: Green (TRUE), Red (FALSE), Yellow (UNDEFINED)
- **Real-time updates** via WebSocket
- **Audio input** with Whisper transcription (optional, requires separate install)
- **Diff view** for disambiguation iterations
- **Cost tracking** with warnings for expensive patterns

## Project Structure

```
logic-answering-system/
├── main.py                 # Entry point
├── requirements.txt        # Dependencies
├── .env.example           # Environment template
├── src/
│   ├── core/
│   │   ├── models.py      # Data models
│   │   ├── cost_tracker.py # Cost tracking
│   │   └── llm.py         # LLM abstraction
│   ├── pipelines/
│   │   ├── disambiguation.py  # Ambiguity lowering
│   │   ├── reasoning.py       # Reasoning loop
│   │   ├── verification.py    # Claim verification
│   │   └── orchestrator.py    # Main coordinator
│   └── ui/
│       └── app.py         # FastAPI web app
├── static/
│   ├── css/style.css
│   └── js/app.js
├── templates/
│   └── index.html
└── logs/                  # Cost logs (JSON)
```

## License

MIT
