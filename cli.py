#!/usr/bin/env python3
"""
Kalshi AI Trading Bot -- Unified CLI (with local Streamlit web dashboard)
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# ... (all your original functions unchanged until cmd_dashboard)

def cmd_dashboard(args: argparse.Namespace) -> None:
    """Launch dashboard — console or web."""
    if getattr(args, "web", False):
        print("🚀 Starting LOCAL Streamlit Web Dashboard at http://localhost:8501")
        import subprocess
        script = Path(__file__).parent / "streamlit_dashboard.py"
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(script)], check=False)
        return

    # Original console dashboard fallback
    import subprocess
    dashboard_script = Path(__file__).parent / "scripts" / "launch_dashboard.py"
    beast_dashboard = Path(__file__).parent / "scripts" / "beast_mode_dashboard.py"
    if dashboard_script.exists():
        subprocess.run([sys.executable, str(dashboard_script)], check=False)
    elif beast_dashboard.exists():
        from src.utils.logging_setup import setup_logging
        from beast_mode_bot import BeastModeBot
        setup_logging(log_level="INFO")
        bot = BeastModeBot(live_mode=False, dashboard_mode=True)
        try:
            asyncio.run(bot.run())
        except KeyboardInterrupt:
            print("\nDashboard stopped by user.")
    else:
        print("Error: No dashboard script found.")
        sys.exit(1)

# ... (rest of your cli.py unchanged)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(...)
    # ... (original subparsers)

    p_dash = subparsers.add_parser("dashboard", help="Launch dashboard")
    p_dash.add_argument("--web", action="store_true", help="Launch beautiful local web dashboard[](http://localhost:8501)")
    p_dash.set_defaults(func=cmd_dashboard)

    return parser

# ... (rest of cli.py unchanged)