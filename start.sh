#!/bin/bash

# Build the custom Triton container
docker build -t triton-ensemble .

# Run the container
docker run --gpus all -p8000:8000 -p8001:8001 -p8002:8002 triton-ensemble 