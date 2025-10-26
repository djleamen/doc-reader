#!/bin/bash
# Start script for the RAG Document Q&A system
# This is a simple wrapper around the main.py entry point

echo "Starting RAG Document Q&A System..."

# Check if virtual environment exists and activate it
if [ -d "venv" ]; then
    echo "Using virtual environment..."
    source venv/bin/activate
    PYTHON_CMD="venv/bin/python"
else
    # Check if Python is available
    if ! command -v python3 &> /dev/null; then
        echo "❌ Python not found. Please install Python 3.8+ and try again."
        exit 1
    fi
    PYTHON_CMD="python3"
fi

# Run the unified entry point
$PYTHON_CMD main.py start "$@"
