"""
Logic Answering System - Main Entry Point

A reasoning system that uses large language models to approach truth
through iterative disambiguation, reasoning, and verification.

Usage:
    # Run the web server
    python main.py serve

    # Process a single question via CLI
    python main.py ask "What is the meaning of life?"

    # Interactive CLI mode
    python main.py interactive
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def serve(host: str = "0.0.0.0", port: int = 8000):
    """Run the web server."""
    from src.ui.app import run_server
    print(f"Starting Logic Answering System on http://{host}:{port}")
    run_server(host=host, port=port)


async def ask_question(question: str, verbose: bool = False):
    """Process a single question."""
    from src.pipelines.orchestrator import process_question_simple

    print(f"\nProcessing: {question}\n")
    print("-" * 60)

    result = await process_question_simple(question)

    print(f"\nOriginal Question: {result['original_question']}")

    if result['refined_question'] != result['original_question']:
        print(f"Refined Question: {result['refined_question']}")

    print(f"\nStatus: {result['status']}")

    if result['final_answer']:
        print(f"\n{'=' * 60}")
        print(f"ANSWER: {result['final_answer']}")
        print(f"{'=' * 60}")
    elif result['termination_reason']:
        print(f"\nResult: {result['termination_reason']}")

    # Cost breakdown
    cost = result['cost_breakdown']
    print(f"\n--- Cost Summary ---")
    print(f"Total: ${cost['total_cost_usd']:.4f}")
    print(f"Opus: ${cost['opus_cost_usd']:.4f} ({cost['opus_calls']} calls)")
    print(f"Flash: ${cost['flash_cost_usd']:.4f} ({cost['flash_calls']} calls)")

    if cost['warnings']:
        print("\nWarnings:")
        for warning in cost['warnings']:
            print(f"  - {warning}")

    if verbose and result['tree_view']:
        print("\n--- Full Tree View ---")
        import json
        print(json.dumps(result['tree_view'], indent=2))


async def interactive_mode():
    """Run in interactive CLI mode."""
    from src.pipelines.orchestrator import Orchestrator
    from src.core.models import SystemState

    print("\n" + "=" * 60)
    print("Logic Answering System - Interactive Mode")
    print("=" * 60)
    print("\nType your question and press Enter. Type 'quit' to exit.\n")

    async def user_callback(query_type: str, question: str) -> str:
        print(f"\n[{query_type.upper()}] {question}")
        response = input("Your response: ")
        return response

    async def state_callback(state: SystemState):
        status = state.context.status
        cost = state.cost_summary.total_cost_usd
        print(f"\r[{status}] Cost: ${cost:.4f}", end="", flush=True)

    orchestrator = Orchestrator(
        user_query_callback=user_callback,
        on_state_update=state_callback,
    )

    while True:
        try:
            question = input("\nQuestion: ").strip()

            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if not question:
                continue

            state = await orchestrator.process_question(question)

            print("\n")

            if state.context.final_answer:
                print(f"{'=' * 60}")
                print(f"ANSWER: {state.context.final_answer}")
                print(f"{'=' * 60}")
            elif state.context.termination_reason:
                print(f"Result: {state.context.termination_reason}")

            print(f"\nTotal cost: ${state.cost_summary.total_cost_usd:.4f}")

            # Offer feedback option
            while True:
                feedback = input("\nProvide feedback (or press Enter to continue): ").strip()
                if not feedback:
                    break

                state = await orchestrator.continue_with_feedback(state, feedback)

                if state.context.final_answer:
                    print(f"\nUPDATED ANSWER: {state.context.final_answer}")
                elif state.context.termination_reason:
                    print(f"\nResult: {state.context.termination_reason}")

        except KeyboardInterrupt:
            print("\nInterrupted. Type 'quit' to exit or continue with a new question.")
        except Exception as e:
            print(f"\nError: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Logic Answering System - Approaching truth through iterative reasoning"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Run the web server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")

    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Process a single question")
    ask_parser.add_argument("question", help="The question to process")
    ask_parser.add_argument("-v", "--verbose", action="store_true", help="Show verbose output")

    # Interactive command
    subparsers.add_parser("interactive", help="Run in interactive CLI mode")

    args = parser.parse_args()

    if args.command == "serve":
        serve(host=args.host, port=args.port)
    elif args.command == "ask":
        asyncio.run(ask_question(args.question, verbose=args.verbose))
    elif args.command == "interactive":
        asyncio.run(interactive_mode())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
