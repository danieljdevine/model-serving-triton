import torch
import torchvision.models as models
import torch.nn as nn
import onnx
import os

class PreprocessModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def forward(self, x):
        # Input is expected to be in range [0, 1]
        x = x / 255.0
        x = (x - self.mean) / self.std
        return x

class PostprocessModel(nn.Module):
    def __init__(self, k=5):
        super().__init__()
        self.k = k

    def forward(self, x):
        # Get top k predictions
        values, indices = torch.topk(x, self.k, dim=1)
        # Stack indices and values
        return torch.stack([indices, values], dim=2)

def download_and_convert_models():
    # Create model repository directories
    os.makedirs("/models/resnet50/1", exist_ok=True)
    os.makedirs("/models/preprocess/1", exist_ok=True)
    os.makedirs("/models/postprocess/1", exist_ok=True)

    # Download and convert ResNet50
    model = models.resnet50(pretrained=True)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)

    # Export ResNet50 to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        "/models/resnet50/1/model.onnx",
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    # Create and export preprocessing model
    preprocess_model = PreprocessModel()
    preprocess_model.eval()
    torch.onnx.export(
        preprocess_model,
        dummy_input,
        "/models/preprocess/1/model.onnx",
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    # Create and export postprocessing model
    postprocess_model = PostprocessModel()
    postprocess_model.eval()
    dummy_logits = torch.randn(1, 1000)  # ResNet50 output size
    torch.onnx.export(
        postprocess_model,
        dummy_logits,
        "/models/postprocess/1/model.onnx",
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print("All models downloaded and converted successfully!")

if __name__ == "__main__":
    download_and_convert_models() 