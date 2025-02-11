import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


# Load the dataset
df = pd.read_csv("heart.csv")
# Handle missing values (numeric columns only)
df.fillna(df.select_dtypes(include=['number']).mean(), inplace=True)
# Convert categorical variables to dummy variables
df = pd.get_dummies(df, drop_first=True)  

# List of numerical features to scale
num_features = ['age', 'cholesterol', 'blood_pressure', 'max_heart_rate', 'resting_bp']
# Ensure the selected columns exist in the dataset
num_features = [col for col in num_features if col in df.columns]
# Standardize numerical features if they exist
if num_features:
    scaler = MinMaxScaler()  # Scale between 0 and 1 instead of standardizing
    df[num_features] = scaler.fit_transform(df[num_features])

# Define features (X) and target (y)
X = df.drop(columns=['target'])  # Drop the target column
y = df['target']  # Define the target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)

model_dt = DecisionTreeClassifier(max_depth=6)  
model_dt.fit(X_train, y_train)

y_pred_lr = model_lr.predict(X_test)
y_pred_dt = model_dt.predict(X_test)

model_rf = RandomForestClassifier(n_estimators=100)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)

model_knn = KNeighborsClassifier(n_neighbors=5)
model_knn.fit(X_train, y_train)
y_pred_knn = model_knn.predict(X_test)

# Function to evaluate model
def evaluate_model(y_test, y_pred):
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))

# Evaluate Logistic Regression
print("Logistic Regression Performance:")
evaluate_model(y_test, y_pred_lr)

# Evaluate Decision Tree
print("Decision Tree Performance:")
evaluate_model(y_test, y_pred_dt)

print("Random Forest Performance:")
evaluate_model(y_test, y_pred_rf)

print("K-Nearest Neighbors Performance:")
evaluate_model(y_test, y_pred_knn)

# Get probabilities for ROC Curve
y_prob_lr = model_lr.predict_proba(X_test)[:, 1]
y_prob_dt = model_dt.predict_proba(X_test)[:, 1]

# Compute ROC curve
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_prob_dt)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, label="Logistic Regression")
plt.plot(fpr_dt, tpr_dt, label="Decision Tree")
plt.plot([0,1], [0,1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
