"""
PulseNet API Server — FastAPI entry point.

Usage:
    python main.py                  # Start on default port 8000
    python main.py --port 8080      # Custom port
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="PulseNet API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on changes")
    args = parser.parse_args()

    print(f"\n⚡ PulseNet API v2.0")
    print(f"   Starting on http://{args.host}:{args.port}")
    print(f"   Docs: http://localhost:{args.port}/docs")
    print("=" * 50)

    uvicorn.run(
        "pulsenet.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
