#!/usr/bin/env python3
"""
Setup script for the RAG Document Q&A system.
"""
import os
import sys
import subprocess
from pathlib import Path


def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True


def setup_environment():
    """Set up the development environment."""
    print("🚀 Setting up RAG Document Q&A System...")
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create virtual environment if it doesn't exist
    if not Path("venv").exists():
        if not run_command("python -m venv venv", "Creating virtual environment"):
            return False
    else:
        print("✅ Virtual environment already exists")
    
    # Determine activation command based on OS
    if os.name == 'nt':  # Windows
        activate_cmd = "venv\\Scripts\\activate"
        pip_cmd = "venv\\Scripts\\pip"
    else:  # Unix/Linux/macOS
        activate_cmd = "source venv/bin/activate"
        pip_cmd = "venv/bin/pip"
    
    # Install requirements
    if not run_command(f"{pip_cmd} install -r requirements.txt", "Installing dependencies"):
        return False
    
    # Create necessary directories
    directories = ["documents", "indexes", "logs", "temp", "backups"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    print("✅ Created necessary directories")
    
    # Copy .env.example to .env if it doesn't exist
    env_file = Path(".env")
    if not env_file.exists():
        if Path(".env.example").exists():
            run_command("cp .env.example .env", "Creating .env file from template")
            print("⚠️ Please edit .env file with your API keys")
        else:
            print("❌ .env.example file not found")
            return False
    else:
        print("✅ .env file already exists")
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Edit .env file with your OpenAI API key")
    print("2. Run './start.sh' to start the system")
    print("3. Or run individual components:")
    print("   - API: python -m src.api")
    print("   - CLI: python -m src.cli --help")
    print("   - Web UI: streamlit run src/streamlit_app.py")
    
    return True


def main():
    """Main setup function."""
    try:
        success = setup_environment()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n❌ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
