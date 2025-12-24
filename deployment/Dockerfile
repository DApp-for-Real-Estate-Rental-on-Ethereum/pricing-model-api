FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY deployment/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY deployment/app.py .

# Copy models directory (including all models)
COPY models/ models/
# Verify production model exists
RUN if [ ! -f models/production/random_forest_tuned.pkl ]; then \
    echo "ERROR: Production model (random_forest_tuned.pkl) not found!" && \
    echo "Available files:" && \
    ls -la models/ && \
    if [ -d models/production ]; then ls -la models/production/; fi && \
    exit 1; \
    else \
    echo "✅ Production model found:" && \
    ls -lh models/production/*.pkl; \
    fi

# Create non-root user for security
RUN useradd -m -u 1000 apiuser && chown -R apiuser:apiuser /app
USER apiuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
