# Use a PyTorch image with CUDA support
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies (for OpenCV and git)
RUN apt-get update && apt-get install -y libgl1-mesa-glx git && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir tqdm ipython easydict tensorboard tabulate scikit-learn pandas opencv-python
RUN pip install --no-cache-dir ffmpeg-python ftfy regex
RUN pip install --no-cache-dir matplotlib


