"""
Root-level entry point for the unified bot orchestrator.

Thin wrapper around src.core.bot_orchestrator so you can run:

    python bot_orchestrator.py --beast --lean
    python bot_orchestrator.py --beast --live-beast
    python bot_orchestrator.py --lean
"""
import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.bot_orchestrator import _main
import asyncio

if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        print("\nOrchestrator stopped.")
