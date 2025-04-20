'''
UI Revenue Prediction Tool

High-Level Goal:
Provide a user interface with the movie revenue prediction tool

Low-Level Description: 1) Create a function to predict movie revenue based on pre-trained model.
                       2) Build a UI with a title, instructions, input, button, and output in Streamlit.
                       3) Run program.

Notes: Final step in Movie Revenue Prediction project. Run with "streamlit run revenuePrediction.py".
'''

import streamlit as st
import torch.nn as nn # Neural Network Library
import torch.nn.functional as F # Helps move data in function.
import torch

class Model(nn.Module):
    def __init__(self, in_features = 3, h1 = 16, h2 = 16, out_features = 1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x

# Function to predict revenue
def predict_revenue(model, budget, runtime, release_month):
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor([[budget, runtime, release_month]], dtype=torch.float32)
        predicted_revenue = model(input_tensor)
        return predicted_revenue.item()

# Load trained model
model = Model()
model.load_state_dict(torch.load('movie_revenue_model.pth'))

# ---- Streamlit UI ----
st.title("Movie Revenue Prediction Tool")

st.write("Enter movie details below to predict its box office revenue:")

# User input fields
budget = st.number_input("Budget ($)", min_value=0, value=50000000, step=1000000)
runtime = st.number_input("Runtime (minutes)", min_value=30, max_value=300, value=120, step=5)
release_month = st.slider("Release Month", 1, 12, 7)

# Predict button
if st.button("Predict Revenue"):
    predicted = predict_revenue(model, budget, runtime, release_month)
    st.success(f"Predicted Revenue: ${predicted:,.2f}")
