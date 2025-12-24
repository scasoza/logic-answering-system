"""
Data models for the Logic Answering System.

These models represent the core data structures used throughout the system,
including verification verdicts, claims, reasoning steps, and context management.
"""

from enum import Enum
from typing import Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


class ModelType(str, Enum):
    """Available model types for different tasks."""
    OPUS = "claude-opus-4-5-20250514"  # Smart model for reasoning & completion check
    FLASH = "gemini-2.0-flash"  # Fast model for other tasks


class CallType(str, Enum):
    """Types of API calls for cost tracking."""
    DISAMBIGUATION_DETERMINE = "disambiguation_determine"
    DISAMBIGUATION_BREAK_APART = "disambiguation_break_apart"
    DISAMBIGUATION_SPECIFICITY = "disambiguation_specificity"
    DISAMBIGUATION_RESPONSE = "disambiguation_response"
    DISAMBIGUATION_COHESIVE = "disambiguation_cohesive"
    REASONING_INITIAL = "reasoning_initial"
    REASONING_CLAIM_EXTRACTION = "reasoning_claim_extraction"
    REASONING_VERIFICATION = "reasoning_verification"
    REASONING_COMPLETION_CHECK = "reasoning_completion_check"


class Verdict(str, Enum):
    """Verification verdicts for claims."""
    TRUE = "TRUE"
    FALSE = "FALSE"
    UNDEFINED = "UNDEFINED"


class UndefinedCategory(str, Enum):
    """Categories for UNDEFINED verdicts."""
    AMBIGUOUS = "AMBIGUOUS"
    EMPIRICALLY_UNKNOWN = "EMPIRICALLY_UNKNOWN"
    AXIOM_DEPENDENT = "AXIOM_DEPENDENT"
    LOGICALLY_UNDECIDABLE = "LOGICALLY_UNDECIDABLE"


class APICallLog(BaseModel):
    """Log entry for a single API call."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model: ModelType
    call_type: CallType
    input_tokens: int
    output_tokens: int
    cost_usd: float
    prompt_preview: str = ""  # First 200 chars of prompt
    response_preview: str = ""  # First 200 chars of response
    duration_ms: int = 0


class CostSummary(BaseModel):
    """Summary of costs for a question."""
    question_id: str
    total_cost_usd: float = 0.0
    total_calls: int = 0
    opus_cost_usd: float = 0.0
    opus_calls: int = 0
    flash_cost_usd: float = 0.0
    flash_calls: int = 0
    disambiguation_iterations: int = 0
    max_verification_depth: int = 0
    warnings: list[str] = Field(default_factory=list)
    calls: list[APICallLog] = Field(default_factory=list)


class ComponentMeaning(BaseModel):
    """A potential meaning for a sentence component."""
    component: str
    meaning: str
    confidence: float = 0.0


class AmbiguityAnalysis(BaseModel):
    """Analysis of ambiguity in a question."""
    is_ambiguous: bool
    components: list[str] = Field(default_factory=list)
    meanings: list[ComponentMeaning] = Field(default_factory=list)
    clarification_needed: list[str] = Field(default_factory=list)


class DisambiguationResult(BaseModel):
    """Result of the disambiguation pipeline."""
    original_question: str
    refined_question: str
    iterations: int = 0
    analysis_history: list[AmbiguityAnalysis] = Field(default_factory=list)
    user_clarifications: list[dict[str, str]] = Field(default_factory=list)


class Claim(BaseModel):
    """A single claim extracted from reasoning."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    source_reasoning_id: str = ""
    verdict: Optional[Verdict] = None
    undefined_category: Optional[UndefinedCategory] = None
    explanation: str = ""
    required_axiom: Optional[str] = None  # For AXIOM_DEPENDENT
    needed_data: Optional[str] = None  # For EMPIRICALLY_UNKNOWN
    undecidability_proof: Optional[str] = None  # For LOGICALLY_UNDECIDABLE


class VerificationNode(BaseModel):
    """A node in the recursive verification tree."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    depth: int
    claim_text: str
    verdict: Verdict
    undefined_category: Optional[UndefinedCategory] = None
    explanation: str
    children: list["VerificationNode"] = Field(default_factory=list)
    is_leaf: bool = False  # True when verification is TRUE or max depth reached


class ClaimVerification(BaseModel):
    """Complete verification result for a claim including recursive checks."""
    claim: Claim
    verification_tree: VerificationNode
    final_verdict: Verdict
    final_explanation: str
    max_depth_reached: int


class ReasoningBlock(BaseModel):
    """A block of reasoning with its claims and verifications."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    reasoning_text: str
    claims: list[Claim] = Field(default_factory=list)
    verifications: list[ClaimVerification] = Field(default_factory=list)
    all_claims_verified_true: bool = False


class CompletionCheckResult(BaseModel):
    """Result of checking if reasoning answers the question."""
    answers_question: bool
    proves_unanswerable: bool = False
    feedback: str = ""
    final_answer: Optional[str] = None


class GlobalContext(BaseModel):
    """Global context maintained across the reasoning process."""
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    original_question: str
    refined_question: str = ""
    disambiguation_context: list[dict[str, Any]] = Field(default_factory=list)
    reasoning_context: list[dict[str, Any]] = Field(default_factory=list)
    reasoning_blocks: list[ReasoningBlock] = Field(default_factory=list)
    current_iteration: int = 0
    status: str = "initialized"
    final_answer: Optional[str] = None
    termination_reason: Optional[str] = None


class UserQuery(BaseModel):
    """User query to clarify ambiguity or provide information."""
    question: str
    context: str = ""
    awaiting_response: bool = True
    response: Optional[str] = None


class SystemState(BaseModel):
    """Overall system state for a question."""
    context: GlobalContext
    cost_summary: CostSummary
    pending_user_queries: list[UserQuery] = Field(default_factory=list)
    is_complete: bool = False
    tree_view: Optional[dict[str, Any]] = None  # For UI rendering


# Update forward references
VerificationNode.model_rebuild()
