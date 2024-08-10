import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import json
from torchsummary import summary
from dnn import *  # Ensure this imports your SimpleNN as definition

# Load data
data_input = torch.load('filtered_output.pt')
data_output = torch.load('filtered_input.pt')

# Convert to NumPy arrays for easier manipulation
data_input_np = data_input.numpy()
data_output_np = data_output.numpy()

# Determine the sizes dynamically
input_size = data_input_np.shape[1]
output_size = data_output_np.shape[1]

# Split into training + validation and test
X_train_val, X_test, y_train_val, y_test = train_test_split(data_input_np, data_output_np, test_size=0.2, random_state=42)

# Split training + validation into training and validation
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# Compute normalization parameters from the training data
input_min = np.min(X_train, axis=0)
input_max = np.max(X_train, axis=0)

output_min = np.min(y_train, axis=0)
output_max = np.max(y_train, axis=0)

# Save normalization parameters
normalization_params = {
    'input_min': input_min.tolist(),
    'input_max': input_max.tolist(),
    'output_min': output_min.tolist(),
    'output_max': output_max.tolist()
}

with open('normalization_params.json', 'w') as f:
    json.dump(normalization_params, f)

# Apply min-max normalization to training, validation, and test data
X_train_normalized = (X_train - input_min) / (input_max - input_min)
X_val_normalized = (X_val - input_min) / (input_max - input_min)
X_test_normalized = (X_test - input_min) / (input_max - input_min)

y_train_normalized = (y_train - output_min) / (output_max - output_min)
y_val_normalized = (y_val - output_min) / (output_max - output_min)
y_test_normalized = (y_test - output_min) / (output_max - output_min)

# Convert back to PyTorch tensors
X_train_tensor = torch.tensor(X_train_normalized, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val_normalized, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_normalized, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_normalized, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_normalized, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_normalized, dtype=torch.float32)

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move data to the device
X_train_tensor = X_train_tensor.to(device)
X_val_tensor = X_val_tensor.to(device)
X_test_tensor = X_test_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)
y_val_tensor = y_val_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)

# Create datasets and DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize parameters
hidden_sizes = [128, 256, 256, 256,  256, 128]
model = LargerNN(input_size, hidden_sizes, output_size).to(device)

# Print model summary
# Ensure to use the correct input size for summary
summary(model, input_size=(input_size,))

# Define the loss function and optimizer
feature_weights = torch.tensor(np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                1.0, 1.0]), dtype=torch.float32).to(device)
criterion = WeightedMSELoss(feature_weights).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Training loop
for epoch in range(500):  # Number of epochs
    model.train()
    train_loss = 0  # Initialize training loss accumulator

    # Training phase
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)  # Ensure data is on the same device as the model
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()  # Accumulate training loss

    # Calculate average training loss for the epoch
    avg_train_loss = train_loss / len(train_loader)

    # Validation phase
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)  # Ensure data is on the same device as the model
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    # Calculate average validation loss for the epoch
    avg_val_loss = val_loss / len(val_loader)

    # Print training and validation loss
    print(f'Epoch {epoch + 1}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

# Testing loop
model.eval()
test_loss = 0
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)  # Ensure data is on the same device as the model
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item()

print(f'Test Loss: {test_loss / len(test_loader):.4f}')

# Save model configuration and state dictionary
model_config = {
    'input_size': input_size,
    'hidden_sizes': hidden_sizes,
    'output_size': output_size
}

# Save configuration to a JSON file
with open('model_config.json', 'w') as f:
    json.dump(model_config, f)

# Save state dictionary
torch.save(model.state_dict(), 'model_state.pth')
