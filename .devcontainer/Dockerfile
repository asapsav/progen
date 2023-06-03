# Use an official Python runtime as a parent image
FROM python:3.9-alpine

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

# Install system level dependencies
RUN apk add --no-cache gcc musl-dev git

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | sh

# Clone the ProGen project
RUN git clone https://github.com/lucidrains/progen.git

# Change to the project directory
WORKDIR /progen

# Install the project dependencies
RUN poetry install

# Install the latest Haiku for mixed precision training
RUN pip install git+https://github.com/deepmind/dm-haiku

# Install the correct CUDA version for GPU training
# Note: This is commented out because Alpine doesn't support CUDA. If you need CUDA support, consider using an Ubuntu-based image.
# RUN pip install --upgrade "jax[cuda111]" -f https://storage.googleapis.com/jax-releases/jax_releases.html

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME ProGen

# Run the application when the container launches
CMD ["poetry", "run", "python", "train.py"]
