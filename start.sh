#!/bin/bash
# Start script for the RAG Document Q&A system

echo "🚀 Starting RAG Document Q&A System..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p documents indexes logs temp backups

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚙️ Creating .env file from template..."
    cp .env.example .env
    echo "⚠️ Please edit .env file with your API keys before running the system"
    exit 1
fi

# Check if OpenAI API key is set
if grep -q "your_openai_api_key_here" .env; then
    echo "⚠️ Please set your OpenAI API key in the .env file"
    exit 1
fi

# Start the system
echo "🌟 Starting API server..."
python -m src.api &
API_PID=$!

# Wait for API to start
sleep 5

echo "🎨 Starting Streamlit interface..."
streamlit run src/streamlit_app.py &
STREAMLIT_PID=$!

echo ""
echo "✅ RAG Document Q&A System is running!"
echo "📊 API Documentation: http://localhost:8000/docs"
echo "🌐 Web Interface: http://localhost:8501"
echo ""
echo "To stop the system, press Ctrl+C"

# Wait for interrupt
trap 'echo "🛑 Stopping services..."; kill $API_PID $STREAMLIT_PID; exit 0' INT
wait
