# NVIDIA Triton Server Setup with Ensemble Model

This repository contains a setup for running NVIDIA Triton Inference Server with an ensemble model configuration using ResNet50.

## Prerequisites

- Docker installed
- NVIDIA Container Toolkit installed
- NVIDIA GPU with appropriate drivers

## Directory Structure

```
.
├── start.sh                 # Startup script for Triton server
├── Dockerfile              # Custom Triton container definition
└── model_repository/       # Model repository directory
    ├── download_models.py  # Script to download and convert models
    ├── preprocess/        # Preprocessing model
    │   └── config.pbtxt
    ├── resnet50/         # ResNet50 model
    │   └── config.pbtxt
    ├── postprocess/      # Postprocessing model
    │   └── config.pbtxt
    └── ensemble/         # Ensemble model configuration
        └── config.pbtxt
```

## Getting Started

1. Make the startup script executable:
   ```bash
   chmod +x start.sh
   ```

2. Run the startup script:
   ```bash
   ./start.sh
   ```

   This will:
   - Build a custom Triton container with PyTorch and other dependencies
   - Download and convert the models inside the container
   - Start the Triton server with the ensemble model

3. The server will be available at:
   - HTTP/REST: localhost:8000
   - gRPC: localhost:8001
   - Metrics: localhost:8002

## Model Architecture

The ensemble model consists of three components:

1. **Preprocessing Model**:
   - Normalizes input images
   - Converts to the format expected by ResNet50
   - Runs on GPU

2. **ResNet50 Model**:
   - Performs the main inference
   - Runs on GPU
   - Outputs 1000 class scores

3. **Postprocessing Model**:
   - Processes the model outputs
   - Returns top 5 predictions with scores
   - Runs on GPU

## Testing the Server

You can test if the server is running by accessing the health endpoint:
```bash
curl -v localhost:8000/v2/health/ready
```

To test the ensemble model with an image:
```bash
curl -X POST localhost:8000/v2/models/ensemble/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [{
      "name": "input",
      "shape": [1, 3, 224, 224],
      "datatype": "FP32",
      "data": [/* your image data here */]
    }]
  }'
```

## Notes

- All models are downloaded and converted when the container starts
- The entire pipeline runs on GPU for maximum performance
- All models support dynamic batching for improved throughput
- The ensemble model automatically handles the flow of data between components