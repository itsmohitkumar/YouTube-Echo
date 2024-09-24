# Use a base Python image
FROM python:3.10

# Set environment variables
ENV LANGCHAIN_TRACING_V2=true
ENV LANGCHAIN_PROJECT=YouTube-Echo
ENV LANGCHAIN_API_KEY=LANGCHAIN_API_KEY

# Set working directory
WORKDIR /app

# Install necessary packages and clean up
RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port that the app runs on
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]
