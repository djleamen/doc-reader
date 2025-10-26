FROM python:3.14-slim

# Set working directory
WORKDIR /app

# Install system dependencies (sorted alphanumerically)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libmagic-dev \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p documents indexes logs temp backups media staticfiles

# Set Django settings module
ENV DJANGO_SETTINGS_MODULE=django_app.settings

# Collect static files for Django and create non-root user for security
RUN python manage.py collectstatic --noinput || true && \
    useradd -m -u 1000 raguser && \
    chown -R raguser:raguser /app

# Switch to non-root user
USER raguser

# Expose ports
EXPOSE 8000 8001 8501

# Health check for Django
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health/ || exit 1

# Default command - Django server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
