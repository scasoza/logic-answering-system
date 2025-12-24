"""
Main Orchestrator for the Logic Answering System.

Coordinates the full pipeline from question input through
disambiguation to reasoning and final answer.
"""

import uuid
from typing import Optional, Callable, Awaitable, Any

from src.core.models import (
    GlobalContext,
    SystemState,
    UserQuery,
    Verdict,
)
from src.core.cost_tracker import CostTracker
from src.core.llm import LLMClient
from src.pipelines.disambiguation import DisambiguationPipeline
from src.pipelines.reasoning import ReasoningPipeline
from src.pipelines.verification import VerificationPipeline


class Orchestrator:
    """
    Main orchestrator that coordinates the entire reasoning process.

    Flow:
    1. Receive question
    2. Run disambiguation pipeline
    3. Run reasoning pipeline (with embedded verification)
    4. Return final result

    The orchestrator manages:
    - Global context
    - Cost tracking
    - User interaction callbacks
    - State for the UI tree view
    """

    def __init__(
        self,
        user_query_callback: Optional[Callable[[str, str], Awaitable[str]]] = None,
        on_state_update: Optional[Callable[[SystemState], Awaitable[None]]] = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            user_query_callback: Async callback for getting user input
                                 Signature: (query_type, question) -> response
            on_state_update: Async callback called when state changes (for UI updates)
        """
        self.user_query_callback = user_query_callback
        self.on_state_update = on_state_update

    async def process_question(
        self,
        question: str,
        question_id: Optional[str] = None,
    ) -> SystemState:
        """
        Process a question through the full pipeline.

        Args:
            question: The user's question
            question_id: Optional ID for tracking (generated if not provided)

        Returns:
            SystemState with full results
        """
        # Initialize tracking
        question_id = question_id or str(uuid.uuid4())
        cost_tracker = CostTracker(question_id=question_id)

        # Initialize context
        context = GlobalContext(
            question_id=question_id,
            original_question=question,
            status="initializing",
        )

        # Create state
        state = SystemState(
            context=context,
            cost_summary=cost_tracker.get_summary(),
        )

        await self._notify_state_update(state)

        try:
            # Initialize LLM client
            llm_client = LLMClient(cost_tracker)

            # Phase 1: Disambiguation
            context.status = "disambiguating"
            await self._notify_state_update(state)

            disambiguation_pipeline = DisambiguationPipeline(
                llm_client=llm_client,
                cost_tracker=cost_tracker,
                user_query_callback=self.user_query_callback,
            )

            disambiguation_result = await disambiguation_pipeline.run(
                question=question,
                global_context=context,
            )

            # Update state after disambiguation
            state.cost_summary = cost_tracker.get_summary()
            state.tree_view = self._build_tree_view(context)
            await self._notify_state_update(state)

            # Phase 2: Reasoning
            context.status = "reasoning"
            await self._notify_state_update(state)

            verification_pipeline = VerificationPipeline(
                llm_client=llm_client,
                cost_tracker=cost_tracker,
            )

            reasoning_pipeline = ReasoningPipeline(
                llm_client=llm_client,
                cost_tracker=cost_tracker,
                verification_pipeline=verification_pipeline,
                user_input_callback=self.user_query_callback,
            )

            completion_result, reasoning_blocks = await reasoning_pipeline.run(
                question=context.refined_question or question,
                global_context=context,
            )

            # Final state update
            state.cost_summary = cost_tracker.get_summary()
            state.tree_view = self._build_tree_view(context)
            state.is_complete = True

            # Save the log
            cost_tracker.save_log()

            await self._notify_state_update(state)

            return state

        except Exception as e:
            context.status = "error"
            context.termination_reason = str(e)
            state.cost_summary = cost_tracker.get_summary()
            state.tree_view = self._build_tree_view(context)

            # Still save log on error
            cost_tracker.save_log()

            await self._notify_state_update(state)
            raise

    async def continue_with_feedback(
        self,
        state: SystemState,
        user_feedback: str,
    ) -> SystemState:
        """
        Continue processing with user feedback.

        Args:
            state: Current system state
            user_feedback: User's response or feedback

        Returns:
            Updated SystemState
        """
        context = state.context

        # Add feedback to context
        context.reasoning_context.append({
            "type": "user_feedback",
            "feedback": user_feedback,
        })

        # Reset status
        context.status = "reasoning"
        state.is_complete = False

        # Create new cost tracker continuing from previous
        cost_tracker = CostTracker(question_id=context.question_id)

        # Copy previous costs (approximation)
        for call in state.cost_summary.calls:
            cost_tracker.calls.append(call)

        # Initialize new LLM client
        llm_client = LLMClient(cost_tracker)

        # Continue reasoning
        verification_pipeline = VerificationPipeline(
            llm_client=llm_client,
            cost_tracker=cost_tracker,
        )

        reasoning_pipeline = ReasoningPipeline(
            llm_client=llm_client,
            cost_tracker=cost_tracker,
            verification_pipeline=verification_pipeline,
            user_input_callback=self.user_query_callback,
        )

        completion_result, reasoning_blocks = await reasoning_pipeline.run(
            question=context.refined_question or context.original_question,
            global_context=context,
        )

        # Update state
        state.cost_summary = cost_tracker.get_summary()
        state.tree_view = self._build_tree_view(context)
        state.is_complete = True

        cost_tracker.save_log()

        await self._notify_state_update(state)

        return state

    async def _notify_state_update(self, state: SystemState):
        """Notify UI of state change."""
        if self.on_state_update:
            await self.on_state_update(state)

    def _build_tree_view(self, context: GlobalContext) -> dict[str, Any]:
        """
        Build a tree structure for the UI.

        Structure:
        - Ambiguity Phase
          - Determine Ambiguity
          - Break Apart
          - Specificity Queries
          - Resolutions
          - Cohesive Update
        - Reasoning Phase
          - Iteration 1
            - Initial Reasoning
            - Claim Extraction
            - Verification
              - Claim 1
                - Recursive Verification
              - Claim 2
                ...
            - Completion Check
          - Iteration 2
            ...
        """
        tree = {
            "question_id": context.question_id,
            "original_question": context.original_question,
            "refined_question": context.refined_question,
            "status": context.status,
            "phases": []
        }

        # Disambiguation phase
        if context.disambiguation_context:
            disambiguation_phase = {
                "name": "Ambiguity Lowering",
                "status": "complete",
                "children": []
            }

            for item in context.disambiguation_context:
                item_type = item.get("type", "unknown")

                if item_type == "specificity_analysis":
                    disambiguation_phase["children"].append({
                        "name": "Specificity Analysis",
                        "type": "specificity",
                        "data": item,
                    })
                elif item_type == "ambiguity_resolution":
                    disambiguation_phase["children"].append({
                        "name": "Ambiguity Resolution",
                        "type": "resolution",
                        "data": item,
                    })
                elif item_type == "cohesive_update":
                    disambiguation_phase["children"].append({
                        "name": "Question Refinement",
                        "type": "cohesive",
                        "from": item.get("from"),
                        "to": item.get("to"),
                    })
                elif item_type == "disambiguation_complete":
                    disambiguation_phase["original"] = item.get("original")
                    disambiguation_phase["refined"] = item.get("refined")
                    disambiguation_phase["iterations"] = item.get("iterations")

            tree["phases"].append(disambiguation_phase)

        # Reasoning phase
        if context.reasoning_blocks:
            reasoning_phase = {
                "name": "Reasoning",
                "status": context.status,
                "children": []
            }

            for i, block in enumerate(context.reasoning_blocks, 1):
                iteration = {
                    "name": f"Iteration {i}",
                    "type": "iteration",
                    "children": [
                        {
                            "name": "Initial Reasoning",
                            "type": "reasoning",
                            "text": block.reasoning_text[:500] + "..." if len(block.reasoning_text) > 500 else block.reasoning_text,
                            "full_text": block.reasoning_text,
                        },
                        {
                            "name": f"Claims ({len(block.claims)})",
                            "type": "claims",
                            "children": []
                        },
                        {
                            "name": "Verification",
                            "type": "verification",
                            "all_true": block.all_claims_verified_true,
                            "children": []
                        }
                    ]
                }

                # Add claims
                for claim in block.claims:
                    iteration["children"][1]["children"].append({
                        "name": claim.text[:50] + "...",
                        "type": "claim",
                        "full_text": claim.text,
                        "verdict": claim.verdict.value if claim.verdict else None,
                        "category": claim.undefined_category.value if claim.undefined_category else None,
                    })

                # Add verifications with recursive structure
                for v in block.verifications:
                    verification_tree = self._build_verification_node(v.verification_tree)
                    verification_tree["claim"] = v.claim.text
                    verification_tree["final_verdict"] = v.final_verdict.value
                    iteration["children"][2]["children"].append(verification_tree)

                reasoning_phase["children"].append(iteration)

            tree["phases"].append(reasoning_phase)

        # Final answer
        if context.final_answer:
            tree["final_answer"] = context.final_answer

        if context.termination_reason:
            tree["termination_reason"] = context.termination_reason

        return tree

    def _build_verification_node(self, node) -> dict[str, Any]:
        """Recursively build verification node for tree view."""
        result = {
            "name": f"Verification (Depth {node.depth})",
            "type": "verification_node",
            "verdict": node.verdict.value,
            "category": node.undefined_category.value if node.undefined_category else None,
            "explanation": node.explanation[:200] + "..." if len(node.explanation) > 200 else node.explanation,
            "full_explanation": node.explanation,
            "is_leaf": node.is_leaf,
            "children": []
        }

        for child in node.children:
            result["children"].append(self._build_verification_node(child))

        return result


async def process_question_simple(question: str) -> dict:
    """
    Simple function to process a question without callbacks.

    Returns a dict with the results.
    """
    orchestrator = Orchestrator()
    state = await orchestrator.process_question(question)

    return {
        "original_question": state.context.original_question,
        "refined_question": state.context.refined_question,
        "status": state.context.status,
        "final_answer": state.context.final_answer,
        "termination_reason": state.context.termination_reason,
        "cost_breakdown": state.cost_summary.model_dump(mode="json"),
        "tree_view": state.tree_view,
    }
