# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in docker
WORKDIR /app

# Install system-level dependencies (including libgomp1)
RUN apt-get update && apt-get install -y libgomp1

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run app.py when the container launches
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
