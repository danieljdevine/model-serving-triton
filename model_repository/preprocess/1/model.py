import numpy as np
import triton_python_backend_utils as pb_utils
import json

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = model_config = json.loads(args['model_config'])
        self.input_name = "input"
        self.output_name = "output"
        
        # Get input and output configuration
        input_config = pb_utils.get_input_config_by_name(model_config, self.input_name)
        output_config = pb_utils.get_output_config_by_name(model_config, self.output_name)
        
        # Convert Triton types to numpy types
        self.input_dtype = pb_utils.triton_string_to_numpy(input_config['data_type'])
        self.output_dtype = pb_utils.triton_string_to_numpy(output_config['data_type'])

    def execute(self, requests):
        responses = []
        
        for request in requests:
            # Get input tensor
            in_tensor = pb_utils.get_input_tensor_by_name(request, self.input_name)
            
            # Convert to numpy array
            img = in_tensor.as_numpy()
            
            # Normalize the image
            img = img.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
            std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
            img = (img - mean) / std
            
            # Create output tensor
            out_tensor = pb_utils.Tensor(self.output_name, img.astype(self.output_dtype))
            
            # Create response
            response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            responses.append(response)
            
        return responses

    def finalize(self):
        pass 