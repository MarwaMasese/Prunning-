# 1. Pruning a CNN Model on MNIST Dataset

## 1.1 Overview
This project demonstrates how to train a Convolutional Neural Network (CNN) on the MNIST dataset using PyTorch and apply L1 unstructured pruning to reduce the model's parameters while maintaining accuracy. The code performs the following steps:

1. Loads and preprocesses the MNIST dataset.
2. Defines a simple CNN model.
3. Trains the model.
4. Evaluates the model before and after pruning.
5. Applies L1 unstructured pruning to the model.
6. Removes pruning reparameterization to make pruning permanent.

## 1.2 Requirements
Make sure you have the following dependencies installed. If you're running this on Google Colab, these libraries are pre-installed:

```bash
pip install torch torchvision
```

## 1.3 Usage

### 1.3.1 Load and Preprocess the Data
The MNIST dataset is loaded and normalized using `torchvision.datasets` and `torchvision.transforms`.

```python
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
```

### 1.3.2 Define the CNN Model
The model consists of two convolutional layers followed by fully connected layers:

```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
```

### 1.3.3 Train the Model
Training is performed using the Adam optimizer and CrossEntropy loss function.

```python
def train(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")
```

### 1.3.4 Evaluate the Model
The model's accuracy on the test dataset is calculated before and after pruning.

```python
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy
```

### 1.3.5 Apply L1 Unstructured Pruning
Pruning is applied to all convolutional and linear layers.

```python
def apply_pruning(model, amount=0.5):
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.l1_unstructured(module, name="bias", amount=amount)
```

### 1.3.6 Remove Pruning Reparameterization
To make pruning permanent, we remove the reparameterization added by `torch.nn.utils.prune`.

```python
def remove_pruning(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.remove(module, "weight")
            prune.remove(module, "bias")
```

### 1.3.7 Running the Script on Google Colab
If you are running this project on Google Colab, follow these steps:

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload your script or copy-paste the code into a new notebook.
3. Ensure you select GPU as the runtime for faster training:
   - Go to `Runtime` > `Change runtime type` > Select `GPU`.
4. Run the notebook cells sequentially.

For local execution, run the script with:

```bash
python script.py
```

The script will train the model for three epochs, evaluate its accuracy, apply pruning, and re-evaluate its accuracy after pruning and after making pruning permanent.

## 1.4 Expected Output
- Training loss decreases over epochs.
- Accuracy before pruning should be high (~98%).
- Accuracy after pruning may decrease slightly.
- Accuracy after making pruning permanent should remain close to post-pruning values.

## 1.5 Conclusion
This project demonstrates how to apply pruning to a PyTorch CNN model to reduce the number of parameters while maintaining classification performance. This technique can be useful for deploying models on resource-constrained devices.

## 1.6 License
This project is released under the MIT License.

