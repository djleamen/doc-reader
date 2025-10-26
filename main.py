"""
Main entry point for the RAG Document Q&A system.
This is the unified entry point for all system operations.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Constants
MANAGE_PY = 'manage.py'


def main():
    """Main entry point with command routing."""
    parser = argparse.ArgumentParser(
        description="RAG Document Q&A System - Unified Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py start                  # Quick start with setup
  python main.py django                 # Start Django web app
  python main.py cli add doc.pdf        # Add document via CLI
  python main.py cli query "question"   # Query via CLI
  python main.py setup                  # First-time setup only
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Start command (recommended for most users)
    start_parser = subparsers.add_parser("start", help="Quick start with automatic setup")
    start_parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    start_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")

    # Django command
    django_parser = subparsers.add_parser("django", help="Start Django web application (setup required)")
    django_parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    django_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")

    # CLI command
    cli_parser = subparsers.add_parser("cli", help="Use the CLI interface")
    cli_parser.add_argument("cli_args", nargs="*", help="CLI arguments")

    # Setup command (first time only)
    subparsers.add_parser("setup", help="First-time environment setup")

    args = parser.parse_args()

    if not args.command:
        # Default to start command if no args given
        args.command = "start"
        args.host = "127.0.0.1"
        args.port = 8000

    if args.command == "start":
        # Quick start with automatic setup
        print("RAG Document Q&A System - Quick Start")
        print("This will set up and start the system automatically.")

        # Check if setup is needed
        if not Path(".env").exists() or not Path("venv").exists():
            print("First time setup required...")
            run_setup()

        # Start Django with setup
        start_django(args.host, args.port, with_setup=True)

    elif args.command == "django":
        # Direct Django start (assumes setup is done)
        start_django(args.host, args.port, with_setup=False)

    elif args.command == "cli":
        from src.cli import main as cli_main
        # Override sys.argv for CLI
        sys.argv = ["cli"] + args.cli_args
        cli_main()

    elif args.command == "setup":
        run_setup()


def run_setup():
    """Run the setup script."""
    print("⚙️ Running first-time setup...")
    try:
        # First try without capturing output to see what happens
        result = subprocess.run([sys.executable, "setup.py"], check=True)
        print("Setup completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Setup failed with exit code {e.returncode}")
        # Try again with captured output to get error details
        try:
            result = subprocess.run([sys.executable, "setup.py"], capture_output=True, text=True, check=False)
            if result.stdout:
                print(f"Output: {result.stdout}")
            if result.stderr:
                print(f"Error: {result.stderr}")
        except Exception:
            pass
        sys.exit(1)
    except Exception as e:
        print(f"❌ Setup failed with unexpected error: {e}")
        sys.exit(1)


def start_django(host="127.0.0.1", port=8000, with_setup=False):
    """Start the Django application."""
    import django
    from django.core.management import execute_from_command_line

    # Set Django settings module
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django_app.settings')

    try:
        django.setup()
    except Exception as django_error:
        print(f"❌ Django setup failed: {django_error}")
        print("Try running: python main.py setup")
        sys.exit(1)

    if with_setup:
        print("🔧 Running Django migrations and collecting static files...")
        try:
            execute_from_command_line([MANAGE_PY, 'makemigrations'])
            execute_from_command_line([MANAGE_PY, 'migrate'])
            execute_from_command_line([MANAGE_PY, 'collectstatic', '--noinput'])
        except Exception as setup_error:
            print(f"⚠️ Setup warning: {setup_error}")

    # Start Django development server
    print(f"Starting Django server at http://{host}:{port}")
    print("Press Ctrl+C to stop the server")
    try:
        execute_from_command_line([
            MANAGE_PY, 'runserver', f'{host}:{port}'
        ])
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
    except Exception as server_error:
        print(f"❌ Server error: {server_error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
