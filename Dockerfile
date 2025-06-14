FROM nvcr.io/nvidia/tritonserver:23.10-py3

# Install PyTorch and other dependencies
RUN pip install torch torchvision onnx

# Copy the model download script
COPY model_repository/download_models.py /download_models.py

# Create an entrypoint script
RUN echo '#!/bin/bash\n\
python /download_models.py\n\
tritonserver --model-repository=/models\n\
' > /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"] 