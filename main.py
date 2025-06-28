"""
Main entry point for the RAG Document Q&A system.
"""
import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))


def main():
    """Main entry point with command routing."""
    parser = argparse.ArgumentParser(
        description="RAG Document Q&A System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py api                    # Start API server
  python main.py ui                     # Start Streamlit UI
  python main.py cli add doc.pdf        # Add document via CLI
  python main.py cli query "question"   # Query via CLI
  python main.py setup                  # Run setup
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # API command
    api_parser = subparsers.add_parser("api", help="Start the API server")
    api_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    api_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    api_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    # UI command
    ui_parser = subparsers.add_parser("ui", help="Start the Streamlit UI")
    
    # CLI command
    cli_parser = subparsers.add_parser("cli", help="Use the CLI interface")
    cli_parser.add_argument("cli_args", nargs="*", help="CLI arguments")
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Run setup script")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == "api":
        import uvicorn
        from src.api import app
        
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            reload=args.reload
        )
    
    elif args.command == "ui":
        import subprocess
        subprocess.run(["streamlit", "run", "src/streamlit_app.py"])
    
    elif args.command == "cli":
        from src.cli import main as cli_main
        # Override sys.argv for CLI
        sys.argv = ["cli"] + args.cli_args
        cli_main()
    
    elif args.command == "setup":
        import subprocess
        subprocess.run([sys.executable, "setup.py"])


if __name__ == "__main__":
    main()
