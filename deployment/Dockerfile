FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies
# gcc/g++ needed for some python extensions if wheels are missing
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY deployment/requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy application code
COPY deployment/ deployment/
COPY models/ models/
COPY seed_target_user.py .
# Seed data file (correct path)
COPY db-seed-extended.sql .

# Create non-root user
RUN useradd -m -u 1000 apiuser && \
    chown -R apiuser:apiuser /app
USER apiuser

# Expose port
EXPOSE 8000

# Health check using curl (faster/lighter than python script)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "deployment.app:app", "--host", "0.0.0.0", "--port", "8000"]
