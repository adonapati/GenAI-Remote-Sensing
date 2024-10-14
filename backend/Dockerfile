# Use the official Python 3.12 image as the parent image
FROM python:3.12.4-slim

# Set the working directory in the container
WORKDIR /app

# Install the system dependencies required for some Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 7080 for the Flask app
EXPOSE 7080

# Define environment variable for Flask app
ENV FLASK_APP=app.py

# Run the application
CMD ["flask", "run", "--host=0.0.0.0", "--port=7080"]
