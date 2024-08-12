# Use the latest Python image from Docker Hub (or you can use the 'python:slim' image)
FROM python:latest
# Set the working directory inside the container
WORKDIR /app
# Copy the requirements file into the container
COPY requirements.txt .
# Copy the Flask application code into the container
COPY Slack.py ./
# Copy the .env file into the container
COPY .env ./.env
# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt
# Expose the port on which your Flask application will listen
EXPOSE 5000
# Define the command to start the Flask application
CMD ["python3", "./Slack.py"]

# Instructions to build and run the Docker container:
# 1. Build the Docker image using the following command:
#    docker build -t bottone <path>
#
# 2. Run the Docker container with the following command:
#    docker run --restart always -p 5000:5000 bottone