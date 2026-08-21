# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies if required (e.g., for some ML libraries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces run as a non-root user for security.
# We create a user with ID 1000 and give them ownership of the /app directory.
RUN useradd -m -u 1000 user \
    && chown -R user:user /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY --chown=user:user . /app/

# Switch to the new non-root user
USER user

# Expose port 7860 (The port Hugging Face expects)
EXPOSE 7860

# Run the FastAPI application using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
