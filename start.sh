#!/bin/bash
# Start script for the RAG Document Q&A system
# This is a simple wrapper around the main.py entry point

echo "🚀 Starting RAG Document Q&A System..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python not found. Please install Python 3.8+ and try again."
    exit 1
fi

# Run the unified entry point
python3 main.py start "$@"
