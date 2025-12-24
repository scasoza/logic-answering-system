"""
Ambiguity Lowering Pipeline.

This pipeline takes a user question and iteratively refines it to reduce ambiguity.
The goal is to ensure the question "pointer" resolves to a single, specific target.

Philosophy:
- Questions are pointers into a model of reality
- The smaller the targeted search, the more specific the pointer must be
- Truth cannot be ambiguous - if a truth claim points to multiple things, it is undefined
"""

import os
import re
import asyncio
from typing import Optional, Callable, Awaitable

from src.core.models import (
    ModelType,
    CallType,
    AmbiguityAnalysis,
    DisambiguationResult,
    ComponentMeaning,
    GlobalContext,
)
from src.core.llm import LLMClient
from src.core.cost_tracker import CostTracker


# Prompts for each step
DETERMINE_AMBIGUITY_PROMPT = """You are analyzing a question for ambiguity.

PHILOSOPHY:
Questions are pointers into a model of reality. A question says "I think there is something shaped like this in this structure and I want to find it." For a question to be answerable with precision, the pointer must resolve to a single target.

TASK:
Analyze the following question and determine if it contains any ambiguity - terms or phrases that could point to multiple different things.

QUESTION: {question}

Respond in this exact format:
AMBIGUOUS: [YES/NO]
REASONING: [Your analysis of why the question is or is not ambiguous]
AMBIGUOUS_ELEMENTS: [Comma-separated list of ambiguous terms/phrases, or NONE]
"""


BREAK_APART_PROMPT = """Break the following sentence into its constituent semantic components.

Each component should be a meaningful unit that could potentially have multiple interpretations.
Focus on nouns, noun phrases, verbs, and qualifiers that carry specific meaning.

SENTENCE: {question}

List each component on a new line, prefixed with "- ".
Example format:
- component 1
- component 2
- component 3
"""


SPECIFICITY_QUERY_SYSTEM = """You are analyzing a single term or phrase for potential multiple meanings.

Your task is to identify ALL possible interpretations of this term in context.
Be thorough - consider technical meanings, colloquial meanings, domain-specific meanings, and edge cases.

Format your response as:
TERM: [the term being analyzed]
MEANINGS:
1. [First possible meaning with brief explanation]
2. [Second possible meaning with brief explanation]
...
MOST_LIKELY: [number of the most likely intended meaning]
NEEDS_CLARIFICATION: [YES/NO]
"""


RESPONSE_SYSTEM = """You are crafting precise definitions to resolve ambiguity.

You have identified terms with multiple possible meanings. Your task is to either:
1. Determine the most likely intended meaning based on context
2. If truly ambiguous, formulate a SPECIFIC clarifying question

RULES:
- Do NOT ask the user to do your job - formulate specific, targeted questions
- Questions should be answerable with a brief response
- Prefer resolving ambiguity yourself when context makes the meaning clear

Format:
TERM: [the ambiguous term]
RESOLUTION: [RESOLVED/NEEDS_USER_INPUT]
DEFINITION: [The precise definition if resolved]
CLARIFYING_QUESTION: [Specific question if needs user input, otherwise N/A]
"""


COHESIVE_QUESTION_SYSTEM = """You are creating a refined, unambiguous version of a question.

Given the original question and the disambiguation results, produce a single cohesive question that:
1. Preserves the original intent
2. Incorporates all resolved definitions
3. Is maximally specific and unambiguous

Output ONLY the refined question, nothing else.
"""


class DisambiguationPipeline:
    """
    Iteratively refines questions to reduce ambiguity.

    The pipeline runs until:
    - The question is determined to be unambiguous
    - Max iterations reached
    - User intervention completes

    Steps per iteration:
    1. Determine if question is ambiguous
    2. Break apart into components
    3. Query specificity for each component (parallel)
    4. Respond to ambiguous terms (parallel)
    5. Create cohesive updated question
    """

    def __init__(
        self,
        llm_client: LLMClient,
        cost_tracker: CostTracker,
        max_iterations: int = 10,
        user_query_callback: Optional[Callable[[str, str], Awaitable[str]]] = None,
    ):
        self.llm = llm_client
        self.cost_tracker = cost_tracker
        self.max_iterations = int(os.getenv("MAX_DISAMBIGUATION_ITERATIONS", max_iterations))
        self.user_query_callback = user_query_callback

    async def run(
        self,
        question: str,
        global_context: GlobalContext,
    ) -> DisambiguationResult:
        """
        Run the full disambiguation pipeline.

        Args:
            question: The original user question
            global_context: The global context to update

        Returns:
            DisambiguationResult with original and refined questions
        """
        result = DisambiguationResult(
            original_question=question,
            refined_question=question,
        )

        current_question = question

        for iteration in range(self.max_iterations):
            self.cost_tracker.increment_disambiguation()

            # Step 1: Determine if ambiguous
            analysis = await self._determine_ambiguity(current_question, global_context)
            result.analysis_history.append(analysis)

            if not analysis.is_ambiguous:
                # Question is clear, we're done
                result.refined_question = current_question
                result.iterations = iteration + 1
                break

            # Step 2: Break apart components
            components = await self._break_apart_components(current_question)
            if not components:
                components = analysis.clarification_needed or [current_question]

            # Step 3: Specificity queries (parallel)
            meanings = await self._query_specificities(components, global_context)
            analysis.meanings = meanings

            # Step 4: Response to ambiguous terms (parallel)
            resolutions, user_queries = await self._resolve_ambiguities(
                meanings, current_question, global_context
            )

            # Handle user queries if any
            if user_queries and self.user_query_callback:
                for term, question_text in user_queries:
                    response = await self.user_query_callback(term, question_text)
                    result.user_clarifications.append({
                        "term": term,
                        "question": question_text,
                        "response": response,
                    })
                    resolutions.append((term, response))

            # Step 5: Create cohesive updated question
            if resolutions:
                current_question = await self._create_cohesive_question(
                    result.original_question,
                    current_question,
                    resolutions,
                    global_context,
                )

            result.iterations = iteration + 1

        result.refined_question = current_question

        # Update global context
        global_context.refined_question = current_question
        global_context.disambiguation_context.append({
            "type": "disambiguation_complete",
            "original": result.original_question,
            "refined": result.refined_question,
            "iterations": result.iterations,
        })

        return result

    async def _determine_ambiguity(
        self,
        question: str,
        global_context: GlobalContext,
    ) -> AmbiguityAnalysis:
        """Step 1: Determine if the question is ambiguous."""
        prompt = DETERMINE_AMBIGUITY_PROMPT.format(question=question)

        # Include global context if available
        context = None
        if global_context.disambiguation_context:
            context = [
                {"role": "assistant", "content": str(global_context.disambiguation_context)}
            ]

        response, _ = await self.llm.call_flash(
            prompt=prompt,
            call_type=CallType.DISAMBIGUATION_DETERMINE,
            context=context,
        )

        # Parse the response
        is_ambiguous = "AMBIGUOUS: YES" in response.upper()
        elements = []

        # Extract ambiguous elements
        match = re.search(r"AMBIGUOUS_ELEMENTS:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        if match and "NONE" not in match.group(1).upper():
            elements = [e.strip() for e in match.group(1).split(",") if e.strip()]

        return AmbiguityAnalysis(
            is_ambiguous=is_ambiguous,
            clarification_needed=elements,
        )

    async def _break_apart_components(self, question: str) -> list[str]:
        """Step 2: Break the question into semantic components."""
        prompt = BREAK_APART_PROMPT.format(question=question)

        response, _ = await self.llm.call_flash(
            prompt=prompt,
            call_type=CallType.DISAMBIGUATION_BREAK_APART,
        )

        # Parse components from response
        components = []
        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("- "):
                components.append(line[2:].strip())

        return components

    async def _query_specificities(
        self,
        components: list[str],
        global_context: GlobalContext,
    ) -> list[ComponentMeaning]:
        """Step 3: Query specificity for each component in parallel."""
        if not components:
            return []

        # Build parallel calls
        calls = []
        for component in components:
            calls.append({
                "model": ModelType.FLASH,
                "prompt": f"Analyze this term/phrase: {component}",
                "call_type": CallType.DISAMBIGUATION_SPECIFICITY,
                "system_prompt": SPECIFICITY_QUERY_SYSTEM,
            })

        # Execute in parallel
        results = await self.llm.call_parallel(calls)

        # Parse results into ComponentMeaning objects
        meanings = []
        for component, (response, _) in zip(components, results):
            needs_clarification = "NEEDS_CLARIFICATION: YES" in response.upper()
            meanings.append(ComponentMeaning(
                component=component,
                meaning=response,
                confidence=0.5 if needs_clarification else 0.9,
            ))

        # Update global context with specificity analysis
        global_context.disambiguation_context.append({
            "type": "specificity_analysis",
            "components": [m.model_dump() for m in meanings],
        })

        return meanings

    async def _resolve_ambiguities(
        self,
        meanings: list[ComponentMeaning],
        question: str,
        global_context: GlobalContext,
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        """Step 4: Resolve ambiguous terms, identify those needing user input."""
        # Filter to only ambiguous components
        ambiguous = [m for m in meanings if m.confidence < 0.9]
        if not ambiguous:
            return [], []

        # Build parallel calls
        calls = []
        for meaning in ambiguous:
            calls.append({
                "model": ModelType.FLASH,
                "prompt": f"""Original question: {question}

Term to resolve: {meaning.component}

Meaning analysis:
{meaning.meaning}

Provide your resolution.""",
                "call_type": CallType.DISAMBIGUATION_RESPONSE,
                "system_prompt": RESPONSE_SYSTEM,
            })

        # Execute in parallel
        results = await self.llm.call_parallel(calls)

        # Parse results
        resolutions = []
        user_queries = []

        for meaning, (response, _) in zip(ambiguous, results):
            if "RESOLUTION: RESOLVED" in response.upper():
                # Extract definition
                match = re.search(r"DEFINITION:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
                if match:
                    resolutions.append((meaning.component, match.group(1).strip()))
            elif "RESOLUTION: NEEDS_USER_INPUT" in response.upper():
                # Extract clarifying question
                match = re.search(r"CLARIFYING_QUESTION:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
                if match and "N/A" not in match.group(1).upper():
                    user_queries.append((meaning.component, match.group(1).strip()))

        # Update global context
        global_context.disambiguation_context.append({
            "type": "ambiguity_resolution",
            "resolutions": resolutions,
            "pending_queries": [{"term": t, "question": q} for t, q in user_queries],
        })

        return resolutions, user_queries

    async def _create_cohesive_question(
        self,
        original_question: str,
        current_question: str,
        resolutions: list[tuple[str, str]],
        global_context: GlobalContext,
    ) -> str:
        """Step 5: Create a cohesive updated question."""
        resolution_text = "\n".join(
            f"- '{term}' means: {definition}"
            for term, definition in resolutions
        )

        prompt = f"""Original question: {original_question}

Current version: {current_question}

Resolved definitions:
{resolution_text}

Create the refined, unambiguous question:"""

        response, _ = await self.llm.call_flash(
            prompt=prompt,
            call_type=CallType.DISAMBIGUATION_COHESIVE,
            system_prompt=COHESIVE_QUESTION_SYSTEM,
        )

        refined = response.strip()

        # Update global context
        global_context.disambiguation_context.append({
            "type": "cohesive_update",
            "from": current_question,
            "to": refined,
        })

        return refined
