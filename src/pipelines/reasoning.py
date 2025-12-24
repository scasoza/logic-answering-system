"""
Reasoning Pipeline with Claim Extraction and Completion Checking.

This pipeline uses adversarial back and forth to approach the truth through
iterative reasoning, claim extraction, verification, and completion checking.

Philosophy:
- Reality exists and we access it through data
- Our model of reality is what we navigate in
- Truth is a pointer that must resolve to a single target in the model
"""

import re
import asyncio
from typing import Optional, Callable, Awaitable

from src.core.models import (
    ModelType,
    CallType,
    Verdict,
    UndefinedCategory,
    Claim,
    ReasoningBlock,
    CompletionCheckResult,
    GlobalContext,
    ClaimVerification,
)
from src.core.llm import LLMClient
from src.core.cost_tracker import CostTracker
from src.pipelines.verification import VerificationPipeline


INITIAL_REASONING_PROMPT = """You are reasoning to find an answer to a question.

PHILOSOPHY:
- There is reality, which we access through data
- We navigate using a model of reality
- Questions are pointers: "I think there is something shaped like this"
- Truth is also a pointer, describing a shape that must resolve to a single target

YOUR TASK:
Reason step by step to find the answer to this question. Show your complete reasoning process.
Be explicit about your assumptions and logical steps.

QUESTION: {question}

{context}

Provide your reasoning:"""


CLAIM_EXTRACTION_SYSTEM = """You are extracting individual claims from a block of reasoning.

A claim is a single atomic statement that can be verified as true or false.
Break down the reasoning into its smallest verifiable units.

RULES:
1. Each claim should be self-contained and verifiable
2. Preserve the logical structure (if A then B becomes two claims)
3. Include implicit assumptions as separate claims
4. Number each claim

FORMAT:
CLAIM 1: [First atomic claim]
CLAIM 2: [Second atomic claim]
...
"""


COMPLETION_CHECK_SYSTEM = """You are checking if a reasoning chain answers the question.

You will see:
1. The original question
2. The reasoning and its verified claims
3. The verification results

DETERMINE:
1. Does this reasoning answer the question? (The logic must lead to a specific answer)
2. Does this reasoning prove the question CANNOT be answered? (With verification)

If the reasoning does NOT yet answer the question, provide specific feedback on:
- What logical gaps remain
- What additional reasoning is needed
- What claims need to be established

FORMAT:
ANSWERS_QUESTION: [YES/NO]
PROVES_UNANSWERABLE: [YES/NO]
FINAL_ANSWER: [The answer if ANSWERS_QUESTION is YES, otherwise N/A]
FEEDBACK: [Specific feedback if answer is NO]
"""


class ReasoningPipeline:
    """
    Iterative reasoning with claim extraction and verification.

    The reasoning loop:
    1. Initial reasoning (Opus)
    2. Claim extraction (Flash)
    3. Claim verification with recursive verification (Flash)
    4. Completion check (Opus)
    5. If not complete, incorporate feedback and continue
    """

    def __init__(
        self,
        llm_client: LLMClient,
        cost_tracker: CostTracker,
        verification_pipeline: VerificationPipeline,
        max_iterations: int = 10,
        user_input_callback: Optional[Callable[[str, str], Awaitable[str]]] = None,
    ):
        self.llm = llm_client
        self.cost_tracker = cost_tracker
        self.verification = verification_pipeline
        self.max_iterations = max_iterations
        self.user_input_callback = user_input_callback

    async def run(
        self,
        question: str,
        global_context: GlobalContext,
    ) -> tuple[CompletionCheckResult, list[ReasoningBlock]]:
        """
        Run the full reasoning pipeline.

        Args:
            question: The refined question to answer
            global_context: Global context for the session

        Returns:
            Tuple of (completion_result, reasoning_blocks)
        """
        reasoning_blocks = []

        for iteration in range(self.max_iterations):
            global_context.current_iteration = iteration + 1

            # Step 1: Initial reasoning
            reasoning_text = await self._initial_reasoning(
                question, global_context, reasoning_blocks
            )

            reasoning_block = ReasoningBlock(reasoning_text=reasoning_text)

            # Step 2: Claim extraction
            claims = await self._extract_claims(reasoning_text)
            reasoning_block.claims = claims

            # Step 3: Claim verification with recursive verification
            verifications = await self.verification.verify_claims(claims)
            reasoning_block.verifications = verifications

            # Check if all claims are verified true
            all_true = all(v.final_verdict == Verdict.TRUE for v in verifications)
            reasoning_block.all_claims_verified_true = all_true

            # Handle undefined claims that need user input
            await self._handle_undefined_claims(verifications, global_context)

            reasoning_blocks.append(reasoning_block)
            global_context.reasoning_blocks.append(reasoning_block)

            # Format verification results for context
            verification_context = self._format_verification_context(verifications)

            # Update reasoning context
            global_context.reasoning_context.append({
                "type": "reasoning_iteration",
                "iteration": iteration + 1,
                "reasoning": reasoning_text[:500],  # Truncate for context
                "claims_count": len(claims),
                "all_verified_true": all_true,
            })

            # Step 4: Completion check (only if all claims are TRUE or we have undefined)
            if all_true or any(v.final_verdict == Verdict.UNDEFINED for v in verifications):
                completion_result = await self._completion_check(
                    question, reasoning_block, verification_context
                )

                if completion_result.answers_question:
                    global_context.status = "completed"
                    global_context.final_answer = completion_result.final_answer
                    global_context.termination_reason = "Question answered"
                    return completion_result, reasoning_blocks

                if completion_result.proves_unanswerable:
                    global_context.status = "unanswerable"
                    global_context.termination_reason = "Proven unanswerable"
                    return completion_result, reasoning_blocks

                # Add feedback to context for next iteration
                if completion_result.feedback:
                    global_context.reasoning_context.append({
                        "type": "completion_feedback",
                        "feedback": completion_result.feedback,
                    })

        # Max iterations reached
        global_context.status = "max_iterations"
        global_context.termination_reason = f"Max iterations ({self.max_iterations}) reached"

        return CompletionCheckResult(
            answers_question=False,
            proves_unanswerable=False,
            feedback="Maximum reasoning iterations reached without conclusive answer.",
        ), reasoning_blocks

    async def _initial_reasoning(
        self,
        question: str,
        global_context: GlobalContext,
        previous_blocks: list[ReasoningBlock],
    ) -> str:
        """Step 1: Generate initial reasoning using Opus."""
        # Build context from previous iterations
        context_parts = []

        if previous_blocks:
            context_parts.append("PREVIOUS REASONING ATTEMPTS:")
            for i, block in enumerate(previous_blocks[-3:], 1):  # Last 3 blocks
                context_parts.append(f"\n--- Attempt {i} ---")
                context_parts.append(block.reasoning_text[:1000])

                if block.verifications:
                    false_claims = [
                        v for v in block.verifications
                        if v.final_verdict == Verdict.FALSE
                    ]
                    if false_claims:
                        context_parts.append("\nFalsified claims:")
                        for v in false_claims:
                            context_parts.append(f"- {v.claim.text}: {v.final_explanation}")

        # Add any feedback from completion checks
        feedback_items = [
            item for item in global_context.reasoning_context
            if item.get("type") == "completion_feedback"
        ]
        if feedback_items:
            context_parts.append("\nFEEDBACK FROM PREVIOUS ATTEMPTS:")
            for item in feedback_items[-2:]:  # Last 2 feedback items
                context_parts.append(item["feedback"])

        context_str = "\n".join(context_parts) if context_parts else ""

        prompt = INITIAL_REASONING_PROMPT.format(
            question=question,
            context=context_str,
        )

        response, _ = await self.llm.call_opus(
            prompt=prompt,
            call_type=CallType.REASONING_INITIAL,
        )

        return response

    async def _extract_claims(self, reasoning_text: str) -> list[Claim]:
        """Step 2: Extract individual claims from reasoning."""
        prompt = f"Extract claims from this reasoning:\n\n{reasoning_text}"

        response, _ = await self.llm.call_flash(
            prompt=prompt,
            call_type=CallType.REASONING_CLAIM_EXTRACTION,
            system_prompt=CLAIM_EXTRACTION_SYSTEM,
        )

        # Parse claims from response
        claims = []
        for line in response.split("\n"):
            line = line.strip()
            match = re.match(r"CLAIM\s*\d+:\s*(.+)", line, re.IGNORECASE)
            if match:
                claim_text = match.group(1).strip()
                claims.append(Claim(text=claim_text))

        return claims

    async def _handle_undefined_claims(
        self,
        verifications: list[ClaimVerification],
        global_context: GlobalContext,
    ):
        """Handle claims that are undefined and may need user input."""
        if not self.user_input_callback:
            return

        for v in verifications:
            if v.final_verdict != Verdict.UNDEFINED:
                continue

            claim = v.claim

            if claim.undefined_category == UndefinedCategory.EMPIRICALLY_UNKNOWN:
                # Ask user for the needed data
                if claim.needed_data:
                    response = await self.user_input_callback(
                        "empirical_data",
                        f"To verify the claim '{claim.text}', we need: {claim.needed_data}\n\nCan you provide this information?"
                    )
                    if response:
                        global_context.reasoning_context.append({
                            "type": "user_provided_data",
                            "claim": claim.text,
                            "data": response,
                        })

            elif claim.undefined_category == UndefinedCategory.AXIOM_DEPENDENT:
                # Ask user about axiom choice
                if claim.required_axiom:
                    response = await self.user_input_callback(
                        "axiom_choice",
                        f"The claim '{claim.text}' depends on assuming: {claim.required_axiom}\n\nShould we assume this axiom? (yes/no)"
                    )
                    if response and response.lower() in ["yes", "y"]:
                        global_context.reasoning_context.append({
                            "type": "axiom_assumed",
                            "claim": claim.text,
                            "axiom": claim.required_axiom,
                        })

    async def _completion_check(
        self,
        question: str,
        reasoning_block: ReasoningBlock,
        verification_context: str,
    ) -> CompletionCheckResult:
        """Step 4: Check if the reasoning answers the question."""
        prompt = f"""QUESTION: {question}

REASONING:
{reasoning_block.reasoning_text}

VERIFICATION RESULTS:
{verification_context}

Does this reasoning answer the question or prove it cannot be answered?"""

        response, _ = await self.llm.call_opus(
            prompt=prompt,
            call_type=CallType.REASONING_COMPLETION_CHECK,
            system_prompt=COMPLETION_CHECK_SYSTEM,
        )

        # Parse the response
        answers = "ANSWERS_QUESTION: YES" in response.upper()
        unanswerable = "PROVES_UNANSWERABLE: YES" in response.upper()

        final_answer = None
        if answers:
            match = re.search(r"FINAL_ANSWER:\s*(.+?)(?=\n[A-Z_]+:|$)", response, re.DOTALL | re.IGNORECASE)
            if match and "N/A" not in match.group(1).upper():
                final_answer = match.group(1).strip()

        feedback = ""
        match = re.search(r"FEEDBACK:\s*(.+?)$", response, re.DOTALL | re.IGNORECASE)
        if match:
            feedback = match.group(1).strip()

        return CompletionCheckResult(
            answers_question=answers,
            proves_unanswerable=unanswerable,
            final_answer=final_answer,
            feedback=feedback,
        )

    def _format_verification_context(
        self,
        verifications: list[ClaimVerification],
    ) -> str:
        """Format verification results for the completion check context."""
        lines = []

        for i, v in enumerate(verifications, 1):
            claim = v.claim
            verdict = v.final_verdict.value

            line = f"{i}. [{verdict}] {claim.text}"

            if v.final_verdict == Verdict.UNDEFINED and claim.undefined_category:
                line += f" ({claim.undefined_category.value})"

            line += f"\n   Explanation: {v.final_explanation[:200]}"

            lines.append(line)

        return "\n".join(lines)
