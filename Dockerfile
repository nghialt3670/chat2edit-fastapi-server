# Use the official Miniconda3 image as the base image
FROM continuumio/miniconda3

# Set the working directory
WORKDIR /home/nghialt/work/chat2edit/chat2edit-fastapi-server

# Copy the environment.yaml file to the working directory
COPY environment.yaml .

# Create the Conda environment from the environment.yaml file
RUN conda env create -f environment.yaml

# Ensure the conda environment is activated and set it as default
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# Copy the application code to the working directory
COPY . .

# Expose the port that FastAPI will run on
EXPOSE 8000

# Set the command to run the FastAPI server
CMD ["conda", "run", "-n", "myenv", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
