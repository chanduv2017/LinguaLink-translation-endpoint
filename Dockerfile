# Use Python 3.12 image as the base
FROM python:3.10.12

# Set the working directory in the container
WORKDIR /code

# Upgrade pip to the latest version
RUN python -m pip install --upgrade pip

# Copy the requirements file into the container at /code/requirements.txt
COPY ./requirements.txt /code/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Create a user with a home directory
RUN useradd -m -d /home/user user

# Switch to the newly created user
USER user

# Set environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory in the user's home directory
WORKDIR $HOME/app

# Copy the current directory contents into the container at the app directory
COPY --chown=user . $HOME/app

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
