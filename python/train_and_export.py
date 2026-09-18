import struct
import torch
import torch.nn as nn
import torch.optim as optin
from torchvision import datasets, transforms


class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

def export_tensor_to_bin(tensor: torch.Tensor, filename:str):
    tensor = tensor.detach().cpu().to(torch.float32)

    if tensor.dim() == 1:
        rows, cols = 1, tensor.shape[0]
    else:
        if tensor.dim == 2:
            rows,cols = tensor.shape[0], tensor.shape[1] 
        else: 
            tensor = tensor.reshape(tensor.shape[0], -1)
            rows, cols = tensor.shape 

    matrix_np = tensor.numpy()

    with open(filename, "wb") as f:
        f.write(struct.pack("<II", rows, cols))
        f.write(matrix_np.tobytes())

    print(f"Exported {filename} ({rows}x{cols})")



if __name__ == "__main__":
    print("Training model : \n")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307, ) , (0.3081,))])
    train_dataset = datasets.MNIST(root = './data', train = True, download = True, transform = transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
     
    model = DigitClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optin.Adam(model.parameters(), lr=0.001)

    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 200 == 0:
            print(f"Batch {batch_idx}/{len(train_loader)} - LossL {loss.item():.4f}")
        
        print("Training done\n Exporting to file\n")


        export_tensor_to_bin(model.fc1.weight, "fc1_w.bin")
        export_tensor_to_bin(model.fc1.bias, "fc1_b.bin")
        export_tensor_to_bin(model.fc2.weight, "fc2_w.bin")
        export_tensor_to_bin(model.fc2.bias, "fc2_b.bin")
        export_tensor_to_bin(model.fc3.weight, "fc3_w.bin")
        export_tensor_to_bin(model.fc3.bias, "fc3_b.bin")


        test_image , label = train_dataset[0]
        export_tensor_to_bin(test_image, "sample_image.bin")
        print(f"Exported sample image. Ground Truth Label: {label}")
        

    