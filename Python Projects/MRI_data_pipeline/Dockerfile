FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Add src to PYTHONPATH
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# Download MNI atlas
RUN mkdir -p /app/MNI_atlas && \
    wget -O /tmp/mni_atlas.zip \
    http://www.bic.mni.mcgill.ca/~vfonov/icbm/2009/mni_icbm152_nlin_sym_09a_nifti.zip && \
    unzip /tmp/mni_atlas.zip -d /app/MNI_atlas && \
    rm /tmp/mni_atlas.zip

# Create directories
RUN mkdir -p /data/input /data/output /data/config

# Copy the docker config from the main folder
COPY docker_config.json /data/config/config.json

# Set entrypoint
ENTRYPOINT ["python", "src/pipeline.py"]
