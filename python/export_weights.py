import os
import struct
import numpy as np
import torch
import torch.nn as nn

class SimpleNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def export_tensor_to_bin(filename: str, tensor: torch.Tensor):
    matrix: np.ndarray = tensor.detach().cpu().numpy().astype(np.float32)
    rows,cols = matrix.shape

    with open(filename, "wb") as f:
        # header that stores nr of rows(int32), nr of cols(int32) and actual matrix 
       # "ii" = 2 integers = 2*4 so 8 bytes
        header = struct.pack("ii", rows, cols)
        f.write(header)
        f.write(matrix.tobytes())

    total_bytes = os.path.getsize(filename)
    print(f"Exported '{filename}' | Shape: ({rows}x{cols}) | Size: {total_bytes / 1024:.2f} KB")

if __name__ == "__main__":
    os.makedirs("weights", exist_ok=True)

    print("Initializing PyTorch Model")
    model = SimpleNeuralNet()
    export_tensor_to_bin("weights/layer1_weights.bin", model.fc1.weight)
    export_tensor_to_bin("weights/layer2_weights.bin", model.fc2.weight)

    print("Model weights exported")