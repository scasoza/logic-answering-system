"""
LLM abstraction layer supporting Claude Opus and Gemini Flash.

Provides a unified interface for making calls to different models,
with automatic cost tracking and response parsing.
"""

import os
import time
import asyncio
from typing import Optional
from dotenv import load_dotenv

import anthropic
from google import genai
from google.genai import types

from .models import ModelType, CallType
from .cost_tracker import CostTracker

load_dotenv()


class LLMClient:
    """
    Unified LLM client for Opus and Flash models.

    Handles API calls, token counting, and cost tracking.
    """

    def __init__(self, cost_tracker: CostTracker):
        self.cost_tracker = cost_tracker

        # Initialize Anthropic client
        self.anthropic = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )

        # Initialize Google GenAI client
        self.google_client = genai.Client(
            api_key=os.getenv("GOOGLE_API_KEY")
        )

    async def call_opus(
        self,
        prompt: str,
        call_type: CallType,
        system_prompt: Optional[str] = None,
        context: Optional[list[dict]] = None,
        max_tokens: int = 4096,
    ) -> tuple[str, dict]:
        """
        Make a call to Claude Opus.

        Args:
            prompt: The user prompt
            call_type: Type of call for cost tracking
            system_prompt: Optional system prompt
            context: Optional conversation context (list of {"role": str, "content": str})
            max_tokens: Maximum tokens in response

        Returns:
            Tuple of (response_text, usage_info)
        """
        start_time = time.time()

        # Build messages
        messages = []
        if context:
            messages.extend(context)
        messages.append({"role": "user", "content": prompt})

        # Make the API call
        response = await asyncio.to_thread(
            self.anthropic.messages.create,
            model=ModelType.OPUS.value,
            max_tokens=max_tokens,
            system=system_prompt or "",
            messages=messages,
        )

        duration_ms = int((time.time() - start_time) * 1000)
        response_text = response.content[0].text

        # Log the call
        self.cost_tracker.log_call(
            model=ModelType.OPUS,
            call_type=call_type,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            prompt_preview=prompt,
            response_preview=response_text,
            duration_ms=duration_ms,
        )

        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }

        return response_text, usage

    async def call_flash(
        self,
        prompt: str,
        call_type: CallType,
        system_prompt: Optional[str] = None,
        context: Optional[list[dict]] = None,
        max_tokens: int = 4096,
    ) -> tuple[str, dict]:
        """
        Make a call to Gemini Flash.

        Args:
            prompt: The user prompt
            call_type: Type of call for cost tracking
            system_prompt: Optional system prompt
            context: Optional conversation context
            max_tokens: Maximum tokens in response

        Returns:
            Tuple of (response_text, usage_info)
        """
        start_time = time.time()

        # Build content parts for Gemini
        contents = []
        if context:
            for msg in context:
                role = "user" if msg["role"] == "user" else "model"
                contents.append(
                    types.Content(
                        role=role,
                        parts=[types.Part.from_text(text=msg["content"])]
                    )
                )

        # Add current prompt
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)]
            )
        )

        # Configure generation
        config = types.GenerateContentConfig(
            system_instruction=system_prompt if system_prompt else None,
            max_output_tokens=max_tokens,
            temperature=0.7,
        )

        # Make the API call
        response = await asyncio.to_thread(
            self.google_client.models.generate_content,
            model=ModelType.FLASH.value,
            contents=contents,
            config=config,
        )

        duration_ms = int((time.time() - start_time) * 1000)
        response_text = response.text

        # Get token counts from usage metadata
        input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0) or 0
        output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0) or 0

        # Log the call
        self.cost_tracker.log_call(
            model=ModelType.FLASH,
            call_type=call_type,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            prompt_preview=prompt,
            response_preview=response_text,
            duration_ms=duration_ms,
        )

        usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

        return response_text, usage

    async def call(
        self,
        model: ModelType,
        prompt: str,
        call_type: CallType,
        system_prompt: Optional[str] = None,
        context: Optional[list[dict]] = None,
        max_tokens: int = 4096,
    ) -> tuple[str, dict]:
        """
        Unified call method that routes to the appropriate model.
        """
        if model == ModelType.OPUS:
            return await self.call_opus(prompt, call_type, system_prompt, context, max_tokens)
        else:
            return await self.call_flash(prompt, call_type, system_prompt, context, max_tokens)

    async def call_parallel(
        self,
        calls: list[dict],
    ) -> list[tuple[str, dict]]:
        """
        Make multiple API calls in parallel.

        Args:
            calls: List of dicts with keys: model, prompt, call_type, system_prompt, context

        Returns:
            List of (response_text, usage_info) tuples in same order as input
        """
        tasks = []
        for call in calls:
            task = self.call(
                model=call["model"],
                prompt=call["prompt"],
                call_type=call["call_type"],
                system_prompt=call.get("system_prompt"),
                context=call.get("context"),
                max_tokens=call.get("max_tokens", 4096),
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        return results
