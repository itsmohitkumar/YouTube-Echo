# Use a base Python image
FROM python:3.9.20-slim

# Set environment variables
ENV LANGCHAIN_TRACING_V2=true
ENV LANGCHAIN_PROJECT=YouTube-Echo

# Accept the LANGCHAIN_API_KEY as a build argument
ARG LANGCHAIN_API_KEY

# Set it as an environment variable
ENV LANGCHAIN_API_KEY=${LANGCHAIN_API_KEY}

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Copy pyproject.toml and poetry.lock first (for caching)
COPY pyproject.toml poetry.lock ./

# Install Python dependencies with Poetry
RUN poetry install --only main

# Copy the entire project code
COPY . .

# Expose the port the app runs on
EXPOSE 5000

# Run the app using Gunicorn in production mode via Poetry
CMD ["poetry", "run", "gunicorn", "-b", "0.0.0.0:5000", "app:app"]
