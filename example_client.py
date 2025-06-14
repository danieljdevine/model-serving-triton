import numpy as np
import requests
import json
from PIL import Image
import io

def prepare_image(image_path):
    # Load and preprocess image
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Resize to match model input
    img = np.array(img).astype(np.float32)  # Convert to float32
    
    # Ensure image is in HWC format (Height, Width, Channels)
    if len(img.shape) == 2:  # Grayscale
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[2] == 4:  # RGBA
        img = img[:, :, :3]  # Remove alpha channel
    
    # Normalize to [0, 1] range
    img = img / 255.0
    
    # Convert to NCHW format (Batch, Channels, Height, Width)
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    return img

def send_request(image_data):
    # Prepare the request payload
    payload = {
        "inputs": [{
            "name": "input",
            "shape": image_data.shape,
            "datatype": "FP32",
            "data": image_data.tolist()
        }]
    }
    
    # Send request to Triton server
    response = requests.post(
        "http://localhost:8000/v2/models/ensemble/infer",
        json=payload
    )
    
    return response.json()

def main():
    # Example usage
    image_path = "example.jpg"  # Replace with your image path
    try:
        # Prepare image
        image_data = prepare_image(image_path)
        
        # Send request
        response = send_request(image_data)
        
        # Process response
        if "outputs" in response:
            output = np.array(response["outputs"][0]["data"])
            output = output.reshape(-1, 2)  # Reshape to [5, 2] array
            
            print("\nTop 5 predictions:")
            print("Class ID | Score")
            print("----------------")
            for class_id, score in output:
                print(f"{int(class_id):8d} | {score:.4f}")
        else:
            print("Error in response:", response)
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 