"""
Claim Verification Pipeline with Recursive Verification.

This module verifies claims extracted from reasoning, with a recursive verification
system that verifies the verifications themselves until reaching undeniable truth.

Philosophy:
- Truth is a pointer that describes a shape in our model of reality
- A claim is TRUE if there is undeniable direct world evidence for it OR if it can be derived from logic
- Truth cannot be ambiguous - if it points to multiple things, it is undefined
"""

import os
import re
import asyncio
from typing import Optional

from src.core.models import (
    ModelType,
    CallType,
    Verdict,
    UndefinedCategory,
    Claim,
    VerificationNode,
    ClaimVerification,
)
from src.core.llm import LLMClient
from src.core.cost_tracker import CostTracker


VERIFICATION_SYSTEM = """You are a rigorous truth verifier. Your task is to determine if a claim is undeniably true, undeniably false, or undefined.

PHILOSOPHY:
- Truth is a pointer that describes a shape in a model of reality
- A claim is TRUE only if there is undeniable direct world evidence for it OR if it can be derived from pure logic
- Truth cannot be ambiguous - if a claim could point to multiple things, it is UNDEFINED
- Dissonance with the broader model doesn't prove falsity, but is a warning sign

VERDICT CRITERIA:
- TRUE: Undeniably true with direct evidence or logical derivation. No reasonable person could dispute this.
- FALSE: Undeniably false with direct counter-evidence or logical contradiction.
- UNDEFINED: Cannot be determined as TRUE or FALSE. Must specify category.

UNDEFINED CATEGORIES:
1. AMBIGUOUS - The claim could mean different things; needs clarification
2. EMPIRICALLY_UNKNOWN - Requires external data the system lacks
3. AXIOM_DEPENDENT - True under some axiom systems, false under others
4. LOGICALLY_UNDECIDABLE - No amount of reasoning can resolve it (must prove this)

FORMAT YOUR RESPONSE EXACTLY AS:
VERDICT: [TRUE/FALSE/UNDEFINED]
CATEGORY: [Only if UNDEFINED: AMBIGUOUS/EMPIRICALLY_UNKNOWN/AXIOM_DEPENDENT/LOGICALLY_UNDECIDABLE]
EXPLANATION: [Your reasoning]
REQUIRED_AXIOM: [Only if AXIOM_DEPENDENT: state the axiom]
NEEDED_DATA: [Only if EMPIRICALLY_UNKNOWN: what data is needed]
UNDECIDABILITY_REASON: [Only if LOGICALLY_UNDECIDABLE: self-reference/infinite-regress/independence-from-axioms]
"""


UNDECIDABILITY_PROOF_SYSTEM = """You are verifying a claim that something is LOGICALLY_UNDECIDABLE.

The verifier claimed undecidability for this reason. You must verify if this reason is valid.

VALID UNDECIDABILITY REASONS:
1. Self-reference/paradox: The claim refers to its own truth value (like "This statement is false")
2. Infinite regress: Establishing the claim requires establishing another claim which requires establishing the first
3. Independence from axioms: The claim is neither provable nor disprovable from the axioms in use

YOUR TASK:
Verify whether the claim actually has the structure claimed. This is a logical verification that can be TRUE or FALSE.

FORMAT:
VERIFICATION: [TRUE/FALSE]
EXPLANATION: [Why the undecidability claim is or is not valid]
"""


RECURSIVE_VERIFICATION_SYSTEM = """You are verifying a verification.

You have access to the context of what was being verified and the verification given.
Your task is to verify that the verification itself is correct.

Apply the same rigorous standards:
- Is the verification's reasoning sound?
- Are its conclusions properly supported?
- Could the verification be wrong?

FORMAT:
VERDICT: [TRUE/FALSE/UNDEFINED]
CATEGORY: [Only if UNDEFINED]
EXPLANATION: [Your analysis of the verification]
"""


class VerificationPipeline:
    """
    Verifies claims with recursive verification until reaching undeniable truth.

    The verification process:
    1. Verify each claim against truth criteria
    2. Recursively verify the verification itself
    3. Continue until reaching a TRUE verification or max depth
    4. Special handling for LOGICALLY_UNDECIDABLE claims (must prove undecidability)
    """

    def __init__(
        self,
        llm_client: LLMClient,
        cost_tracker: CostTracker,
        max_depth: int = 5,
    ):
        self.llm = llm_client
        self.cost_tracker = cost_tracker
        self.max_depth = int(os.getenv("MAX_VERIFICATION_DEPTH", max_depth))

    async def verify_claims(
        self,
        claims: list[Claim],
        parent_context: Optional[list[dict]] = None,
    ) -> list[ClaimVerification]:
        """
        Verify multiple claims in parallel.

        Args:
            claims: List of claims to verify
            parent_context: Optional context from the verification chain above

        Returns:
            List of ClaimVerification results
        """
        # Run initial verifications in parallel
        tasks = [
            self._verify_single_claim(claim, parent_context)
            for claim in claims
        ]
        results = await asyncio.gather(*tasks)
        return results

    async def _verify_single_claim(
        self,
        claim: Claim,
        parent_context: Optional[list[dict]] = None,
    ) -> ClaimVerification:
        """Verify a single claim with recursive verification."""
        # Initial verification
        root_node = await self._create_verification_node(
            claim_text=claim.text,
            depth=0,
            parent_context=parent_context,
        )

        # Recursive verification until we hit TRUE or max depth
        await self._recursive_verify(root_node, parent_context)

        # Determine final verdict based on the verification tree
        final_verdict, final_explanation = self._determine_final_verdict(root_node)

        # Update claim with verification results
        claim.verdict = final_verdict
        claim.explanation = final_explanation

        if root_node.undefined_category:
            claim.undefined_category = root_node.undefined_category

        return ClaimVerification(
            claim=claim,
            verification_tree=root_node,
            final_verdict=final_verdict,
            final_explanation=final_explanation,
            max_depth_reached=self._get_max_depth(root_node),
        )

    async def _create_verification_node(
        self,
        claim_text: str,
        depth: int,
        parent_context: Optional[list[dict]] = None,
        is_meta_verification: bool = False,
    ) -> VerificationNode:
        """Create a verification node by calling the LLM."""
        self.cost_tracker.update_verification_depth(depth)

        system = RECURSIVE_VERIFICATION_SYSTEM if is_meta_verification else VERIFICATION_SYSTEM
        prompt = f"Verify this claim:\n\n{claim_text}"

        response, _ = await self.llm.call_flash(
            prompt=prompt,
            call_type=CallType.REASONING_VERIFICATION,
            system_prompt=system,
            context=parent_context,
        )

        # Parse the response
        verdict, category, explanation, extra = self._parse_verification_response(response)

        node = VerificationNode(
            depth=depth,
            claim_text=claim_text,
            verdict=verdict,
            undefined_category=category,
            explanation=explanation,
        )

        # Handle undecidability proofs
        if category == UndefinedCategory.LOGICALLY_UNDECIDABLE:
            undecidability_reason = extra.get("undecidability_reason", "")
            if undecidability_reason:
                # Verify the undecidability claim itself
                proof_valid = await self._verify_undecidability_proof(
                    claim_text, undecidability_reason
                )
                if not proof_valid:
                    # Undecidability claim failed verification
                    node.explanation += "\n[NOTE: Undecidability proof was not validated]"

        return node

    async def _recursive_verify(
        self,
        node: VerificationNode,
        parent_context: Optional[list[dict]] = None,
    ):
        """Recursively verify a verification node."""
        # Stop conditions
        if node.depth >= self.max_depth:
            node.is_leaf = True
            return

        if node.verdict == Verdict.TRUE:
            # True verification - still verify it once more to be sure
            if node.depth < self.max_depth - 1:
                # Create meta-verification
                meta_claim = f"""The following verification was made:

Claim: {node.claim_text}
Verdict: {node.verdict.value}
Explanation: {node.explanation}

Is this verification correct?"""

                child_context = parent_context or []
                child_context = child_context + [
                    {"role": "assistant", "content": f"Verified: {node.claim_text} -> {node.verdict.value}"}
                ]

                child = await self._create_verification_node(
                    claim_text=meta_claim,
                    depth=node.depth + 1,
                    parent_context=child_context,
                    is_meta_verification=True,
                )
                node.children.append(child)

                # If meta-verification is TRUE, we're done
                if child.verdict == Verdict.TRUE:
                    child.is_leaf = True
                else:
                    # Continue recursing
                    await self._recursive_verify(child, child_context)
            else:
                node.is_leaf = True
        else:
            # Non-TRUE verdict - mark as leaf, no further verification needed
            node.is_leaf = True

    async def _verify_undecidability_proof(
        self,
        original_claim: str,
        undecidability_reason: str,
    ) -> bool:
        """Verify that an undecidability claim is valid."""
        prompt = f"""Original claim: {original_claim}

Claimed reason for undecidability: {undecidability_reason}

Verify if this claim actually has the structure that makes it undecidable."""

        response, _ = await self.llm.call_flash(
            prompt=prompt,
            call_type=CallType.REASONING_VERIFICATION,
            system_prompt=UNDECIDABILITY_PROOF_SYSTEM,
        )

        return "VERIFICATION: TRUE" in response.upper()

    def _parse_verification_response(
        self,
        response: str,
    ) -> tuple[Verdict, Optional[UndefinedCategory], str, dict]:
        """Parse the verification response into structured data."""
        verdict = Verdict.UNDEFINED
        category = None
        explanation = ""
        extra = {}

        # Parse verdict
        if "VERDICT: TRUE" in response.upper():
            verdict = Verdict.TRUE
        elif "VERDICT: FALSE" in response.upper():
            verdict = Verdict.FALSE
        else:
            verdict = Verdict.UNDEFINED

        # Parse category if undefined
        if verdict == Verdict.UNDEFINED:
            for cat in UndefinedCategory:
                if f"CATEGORY: {cat.value}" in response.upper():
                    category = cat
                    break

        # Parse explanation
        match = re.search(r"EXPLANATION:\s*(.+?)(?=\n[A-Z_]+:|$)", response, re.DOTALL | re.IGNORECASE)
        if match:
            explanation = match.group(1).strip()

        # Parse extra fields
        if category == UndefinedCategory.AXIOM_DEPENDENT:
            match = re.search(r"REQUIRED_AXIOM:\s*(.+?)(?=\n|$)", response, re.IGNORECASE)
            if match:
                extra["required_axiom"] = match.group(1).strip()

        if category == UndefinedCategory.EMPIRICALLY_UNKNOWN:
            match = re.search(r"NEEDED_DATA:\s*(.+?)(?=\n|$)", response, re.IGNORECASE)
            if match:
                extra["needed_data"] = match.group(1).strip()

        if category == UndefinedCategory.LOGICALLY_UNDECIDABLE:
            match = re.search(r"UNDECIDABILITY_REASON:\s*(.+?)(?=\n|$)", response, re.IGNORECASE)
            if match:
                extra["undecidability_reason"] = match.group(1).strip()

        return verdict, category, explanation, extra

    def _determine_final_verdict(
        self,
        root: VerificationNode,
    ) -> tuple[Verdict, str]:
        """Determine the final verdict from the verification tree."""
        # Find the deepest TRUE node
        deepest_true = self._find_deepest_true(root)

        if deepest_true:
            return Verdict.TRUE, f"Verified TRUE at depth {deepest_true.depth}: {deepest_true.explanation}"

        # Otherwise, return the root verdict
        return root.verdict, root.explanation

    def _find_deepest_true(self, node: VerificationNode) -> Optional[VerificationNode]:
        """Find the deepest TRUE verification node."""
        if not node.children:
            return node if node.verdict == Verdict.TRUE else None

        deepest = None
        for child in node.children:
            child_deepest = self._find_deepest_true(child)
            if child_deepest:
                if not deepest or child_deepest.depth > deepest.depth:
                    deepest = child_deepest

        if not deepest and node.verdict == Verdict.TRUE:
            return node

        return deepest

    def _get_max_depth(self, node: VerificationNode) -> int:
        """Get the maximum depth reached in the verification tree."""
        if not node.children:
            return node.depth

        max_child = max(self._get_max_depth(child) for child in node.children)
        return max(node.depth, max_child)

    def format_verification_tree(self, root: VerificationNode, indent: int = 0) -> str:
        """Format a verification tree as a string for context."""
        lines = []
        prefix = "  " * indent

        verdict_symbol = {
            Verdict.TRUE: "✓",
            Verdict.FALSE: "✗",
            Verdict.UNDEFINED: "?",
        }

        symbol = verdict_symbol.get(root.verdict, "?")
        category_str = f" [{root.undefined_category.value}]" if root.undefined_category else ""

        lines.append(f"{prefix}{symbol} Depth {root.depth}: {root.verdict.value}{category_str}")
        lines.append(f"{prefix}  Claim: {root.claim_text[:100]}...")
        lines.append(f"{prefix}  Explanation: {root.explanation[:150]}...")

        for child in root.children:
            lines.append(self.format_verification_tree(child, indent + 1))

        return "\n".join(lines)
