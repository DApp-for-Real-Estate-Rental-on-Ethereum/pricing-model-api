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

# Copy entire deployment package (including routers)
COPY deployment/ deployment/
# Copy models directory
COPY models/ models/

# Verify production model exists (compressed .pkl.gz format)
RUN if [ ! -f models/production/random_forest_tuned.pkl.gz ]; then \
    echo "ERROR: Production model (random_forest_tuned.pkl.gz) not found!" && \
    echo "Available files:" && \
    ls -la models/ && \
    if [ -d models/production ]; then ls -la models/production/; fi && \
    exit 1; \
    else \
    echo "✅ Production model found:" && \
    ls -lh models/production/random_forest_tuned.pkl.gz; \
    fi

# Create non-root user for security
RUN useradd -m -u 1000 apiuser && chown -R apiuser:apiuser /app
USER apiuser

# Expose port
EXPOSE 8000

# Health check (adjust URL to match new path structure if needed, but port is same)
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

# Run the application (pointing to module path)
CMD ["uvicorn", "deployment.app:app", "--host", "0.0.0.0", "--port", "8000"]
