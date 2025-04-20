'''
Movie Revenue Prediction Tool

High-Level Goal:
Leverage public data on popular cinematic films to train a machine learning model to accept inputs of:
    1) Title,
    2) Budget,
    3) Runtime,
    4) Genres,
    5) Release_date,
    6) Production_companies,
    7) Director.
And return an output of:
    1) Predicted Movie Revenue.

Low-Level Description: 1) Load collected movie data (3930 cleaned entries).
                       2) Convert dates to months.
                       3) Assign weights.
                       4) Train model.

Notes: Second step in Movie Revenue Prediction project. After successfully training the model,
       to predict a movie revenue given my inputs, I will create a user interface for ease of accessibility.
'''

import torch # PyTorch
import torch.nn as nn # Neural Network Library
import torch.nn.functional as F # Helps move data in function.
import torch.optim as optim  # For optimization.
import pandas as pd # Load data.
from sklearn.model_selection import train_test_split # Train split.

class Model(nn.Module):
    # Features: Budget, Runtime, Release Month.
    # 16 Neurons Arbitrary because they fit nicely on a graphic.
    # 1 Out feature for 1 revenue prediction.
    def __init__(self, in_features = 3, h1 = 16, h2 = 16, out_features = 1):
        super().__init__() # Instantiate nn.Module
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    # Pushes x forward through the neural network.
    def forward(self, x):
        x = F.relu(self.fc1(x)) # Rectificed Linear Unit, if output < 0: Set as 0; Else use output.
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x

# Load data, for now stick to numbers: Revenue, Runtime, Month (0 - 12)
df = pd.read_csv("movie_dataset.csv")
df['release_month'] = pd.to_datetime(df['release_date']).dt.month

# Train Test Split
x = df[['budget', 'runtime', 'release_month']] # 2 dimensions for inputs
y = df['revenue'] # 1 dimension for output

# Convert to numPy arrays
x = x.values
y = y.values

# Convert to torch tensors
X = torch.tensor(x, dtype=torch.float32)
Y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Train/test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42) # 80/20 train/test split

# 1. Initialize Model, Loss, and Optimizer
model = Model()
criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Learning rate

# 2. Training Parameters
epochs = 100
batch_size = 32  # Optional: For mini-batch training

# 3. Training Loop
for epoch in range(epochs):
    model.train()  # Set model to training mode
    optimizer.zero_grad()  # Clear gradients
    
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, Y_train)
    
    # Backward pass and optimize
    loss.backward()
    optimizer.step()
    
    # Print training progress
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

model.eval()  # Set model to evaluation mode
with torch.no_grad():  # Disable gradient tracking
    test_outputs = model(X_test)
    test_loss = criterion(test_outputs, Y_test)
    print(f'Test Loss (MSE): {test_loss.item():.4f}')

# Save the Trained Model
torch.save(model.state_dict(), 'movie_revenue_model.pth')
