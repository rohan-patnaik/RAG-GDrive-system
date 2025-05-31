# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PIP_NO_CACHE_DIR off
ENV PIP_DISABLE_PIP_VERSION_CHECK on

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (if any, e.g., for certain Python packages)
# RUN apt-get update && apt-get install -y --no-install-recommends some-package && rm -rf /var/lib/apt/lists/*

# Create a non-root user and group
RUN groupadd -r appgroup && useradd --no-log-init -r -g appgroup appuser

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# RUN pip install --upgrade pip # Not always needed with slim-buster
RUN pip install -r requirements.txt

# Copy the rest of the application code
# We copy backend into /app/backend to match PYTHONPATH expectations
COPY ./backend /app/backend
COPY ./data /app/data 
# Copy initial data if needed, though volumes are preferred for persistence
# Ensure scripts are executable if needed by CMD or ENTRYPOINT
COPY ./scripts/wait_for_system.py /app/rag_system/utils/wait_for_system.py
RUN chmod +x /app/rag_system/utils/wait_for_system.py


# Set PYTHONPATH to include the backend directory where rag_system package resides
ENV PYTHONPATH "${PYTHONPATH}:/app/backend"

# Create necessary directories and set permissions
# These directories will be mounted as volumes in docker-compose,
# but creating them here ensures they exist if not mounted.
RUN mkdir -p /app/data/vector_store /app/logs && \
    chown -R appuser:appgroup /app/data /app/logs

# Switch to the non-root user
USER appuser

# Expose the port the app runs on (should match .env and uvicorn command)
# This is informational; actual port mapping is done in docker-compose.yml or `docker run -p`
# EXPOSE 8000 # This will be dynamically set by API_PORT in docker-compose

# Command to run the application
# The actual command is specified in docker-compose.yml for flexibility
# CMD ["uvicorn", "rag_system.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "backend/"]
# A placeholder CMD or ENTRYPOINT is good practice
CMD ["echo", "Default command: Use docker-compose to run the application with specific parameters."]
