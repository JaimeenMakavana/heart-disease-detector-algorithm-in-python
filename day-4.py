import torch
import torch.nn as nn
import torch.optim as optim
import random

# Step 1: Define Chain of Thought Model (LSTM-based model for cricket predictions)
class ChainOfThoughtModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChainOfThoughtModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take the last LSTM output
        return out

# Step 2: Define Reward Function (Reinforcement Learning - RL)
def reward_function(prediction, target):
    return -torch.abs(prediction - target).mean()  # Negative error as reward

# Step 3: Define Distilled Model (Simpler model for fast inference)
class DistilledModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DistilledModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# Step 4: Generate Cricket Data (Simulating Runs based on balls faced, strike rate, average runs)
def generate_cricket_data(num_samples=1000):
    X, y = [], []
    for _ in range(num_samples):
        balls_faced = random.randint(10, 100)
        strike_rate = random.uniform(50, 150)  # Strike rate between 50 and 150
        avg_runs_last_5 = random.randint(10, 80)  # Average runs in last 5 innings
        runs_scored = int((balls_faced * strike_rate) / 100 + avg_runs_last_5 * 0.5)  # Approximate logic
        
        X.append([balls_faced, strike_rate, avg_runs_last_5])
        y.append([runs_scored])
    
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Step 5: Training the Models
def train_models():
    input_size, hidden_size, output_size = 3, 16, 1  # 3 inputs, 1 output
    cot_model = ChainOfThoughtModel(input_size, hidden_size, output_size)
    student_model = DistilledModel(input_size, hidden_size, output_size)
    
    optimizer = optim.Adam(cot_model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    
    X_train, y_train = generate_cricket_data(500)
    X_train = X_train.unsqueeze(1)  # Reshaping for LSTM
    
    for epoch in range(100):
        optimizer.zero_grad()
        predictions = cot_model(X_train)
        loss = -reward_function(predictions, y_train)  # RL-based loss
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
    
    # Distillation Phase
    optimizer_student = optim.Adam(student_model.parameters(), lr=0.01)
    for epoch in range(50):
        optimizer_student.zero_grad()
        student_predictions = student_model(X_train.squeeze(1))
        teacher_predictions = cot_model(X_train).detach()
        distillation_loss = loss_fn(student_predictions, teacher_predictions)
        distillation_loss.backward()
        optimizer_student.step()
        if epoch % 10 == 0:
            print(f"Distillation Epoch {epoch}, Loss: {distillation_loss.item()}")
    
    return cot_model, student_model

# Train and test the models
cot_model, distilled_model = train_models()

# Example prediction
def predict_runs(balls_faced, strike_rate, avg_runs_last_5):
    input_tensor = torch.tensor([[balls_faced, strike_rate, avg_runs_last_5]], dtype=torch.float32)
    input_tensor = input_tensor.unsqueeze(1)
    
    with torch.no_grad():
        prediction = cot_model(input_tensor)
    return prediction.item()

# Test Prediction
runs = predict_runs(50, 120, 40)  
print(f"Predicted Runs: {runs}")
