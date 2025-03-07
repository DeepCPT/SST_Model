import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.stats import norm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# 1. Generate data from Reserved Utility function from Kim et al. (2010)
def generate_data(num_samples=25000):
    R = np.random.uniform(low=-1, high=2, size=num_samples) # Reservation Utility input
    V= np.random.uniform(low=-1, high=2, size=num_samples)  # Known proportion Utility input

    PDF=norm.pdf(R-V)
    CDF=norm.cdf(R-V)
    C=(1-CDF)*((V-R)+PDF/(1-CDF)) # c should be positive

    #result = np.column_stack((R, V, C))

    X = np.column_stack((V, C))
    y = R

    # Identify rows without NaN in arr1
    valid_rows = ~np.isnan(X).any(axis=1)

    # Filter arr1 and arr2 based on valid rows
    X = X[valid_rows]
    y = y[valid_rows]

    return X, y, np.column_stack((V, R))

# 2. Create a simple feedforward neural network
class PreTrainedModuleRU(nn.Module):
    def __init__(self):
        super(PreTrainedModuleRU, self).__init__()
        self.fc1 = nn.Linear(2, 16)  # Input layer to hidden layer
        self.fc2 = nn.Linear(16, 10)   # Hidden layer
        self.fc3 = nn.Linear(10, 8)  # Hidden layer
        self.fc4 = nn.Linear(8, 1)     # Hidden layer to output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Activation function
        x = torch.relu(self.fc2(x))  # Activation function
        x = torch.relu(self.fc3(x))  # Activation function
        x = self.fc4(x)               # Output layer
        return x

# 3. Prepare the data
X, y, data_raw= generate_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)  # Reshape to (num_samples, 1)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)  # Reshape to (num_samples, 1)

# 4. Set up the model, loss function, and optimizer
model = PreTrainedModuleRU()
criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam optimizer

# 5. Train the model
num_epochs = 5000
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    optimizer.zero_grad()  # Zero the gradients
    outputs = model(X_train_tensor)  # Forward pass
    loss = criterion(outputs, y_train_tensor)  # Compute the loss
    loss.backward()  # Backward pass
    optimizer.step()  # Update weights

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

# 6. Save the model to a .pt file
# model_path = 'model_ru_parameter.pt'
# torch.save(model.state_dict(), model_path)
# print(f'Model saved to {model_path}')

# 7. Evaluate the model
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    predicted = model(X_test_tensor)  # Get predictions on the test set
    test_loss = criterion(predicted, y_test_tensor)  # Calculate the test loss
    print(f'Test Loss: {test_loss.item():.6f}')

# 8. Plot the results


# Create scatter plots for two datasets
fig = go.Figure()

# Dataset 1 with a fixed color (e.g., blue)
fig.add_trace(go.Scatter3d(
    x=X_test[0:50, 0], y=X_test[0:50, 1], z=y_test[0:50],
    mode='markers',
    marker=dict(size=5, color='blue',opacity=0.5),
    name='Dataset 1'
))

# Dataset 2 with a fixed color (e.g., red)
pred=predicted.numpy()
fig.add_trace(go.Scatter3d(
    x=X_test[0:50, 0], y=X_test[0:50, 1], z=pred[0:50],
    mode='markers',
    marker=dict(size=5, color='red',opacity=0.5),
    name='Dataset 2'
))

# Update layout for better visuals
fig.update_layout(
    scene=dict(
        xaxis_title='X-axis',
        yaxis_title='Y-axis',
        zaxis_title='Z-axis'
    ),
    margin=dict(l=0, r=0, b=0, t=0)
)

# Show interactive plot
fig.show()

