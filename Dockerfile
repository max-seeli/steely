# Use an official Python image as the base
FROM python:3.12-slim-bookworm

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Copy the project files into the container
ADD . /app

# Set the working directory
WORKDIR /app


# Install Python dependencies
RUN uv sync --locked

# Install nltk data for offline use
RUN uv run src/steely/nltk_loader.py

# Set the entrypoint to the inference script
ENTRYPOINT ["/app/.venv/bin/python3", "/app/src/steely/task_1/correlation_signal_classifier.py"]
