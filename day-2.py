import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score , roc_auc_score, log_loss, matthews_corrcoef, confusion_matrix

# Load the dataset
df = pd.read_csv("heart.csv")
# Handle missing values (numeric columns only)
df.fillna(df.select_dtypes(include=['number']).mean(), inplace=True)
# Convert categorical variables to dummy variables
df = pd.get_dummies(df, drop_first=True)  

# List of numerical features to scale
num_features = ["age","sex","cp","trestbps"]
# num_features = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","target"]
# Ensure the selected columns exist in the dataset
num_features = [col for col in num_features if col in df.columns]
# Standardize numerical features if they exist
if num_features:
    scaler = MinMaxScaler()  # Scale between 0 and 1 instead of standardizing
    df[num_features] = scaler.fit_transform(df[num_features])

# Define features (X) and target (y)
X = df.drop(columns=['target'])  # Drop the target column
y = df['target']  # Define the target
# Undersample the majority class (reduce healthy patients)
# df_healthy = df[df["target"] == 0].sample(n=30, random_state=42)  # Reduce majority class
# df_disease = df[df["target"] == 1]  # Keep all cases of disease
# df_imbalanced = pd.concat([df_healthy, df_disease])

# Redefine features (X) and target (y)
# X = df_imbalanced.drop(columns=['target'])
# y = df_imbalanced['target']
# X_noisy = X + np.random.normal(0, 5, X.shape)
# random state: shuffling card in same direction

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)
print('model_lr: ', model_lr)

# detective questions: depth theory -> right balance
model_dt = DecisionTreeClassifier(max_depth=5)  
model_dt.fit(X_train, y_train)

y_pred_dt = model_dt.predict(X_test)

model_rf = RandomForestClassifier(n_estimators=100)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)

model_knn = KNeighborsClassifier(n_neighbors=5)
model_knn.fit(X_train, y_train)
y_pred_knn = model_knn.predict(X_test)

# print(y_pred_lr, y_pred_dt, y_pred_rf, y_pred_knn)  => 50 elements of array
# Function to evaluate model
def evaluate_model(y_test, y_pred):
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, zero_division=1))
    # print("Recall:", recall_score(y_test, y_pred, zero_division=1))
    # print("F1 Score:", f1_score(y_test, y_pred, zero_division=1))
    # print("ROC AUC Score:", roc_auc_score(y_test, y_pred))
    # print("Log Loss:", log_loss(y_test, y_pred))
    # print("Matthews Correlation Coefficient:", matthews_corrcoef(y_test, y_pred))
    # print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# print("y-test",(y_test))
# print("Logistic Regression Performance:")
evaluate_model(y_test, y_pred_lr)

# print("Decision Tree Performance:")
evaluate_model(y_test, y_pred_dt)

# print("Random Forest Performance:")
evaluate_model(y_test, y_pred_rf)

# print("K-Nearest Neighbors Performance:")
evaluate_model(y_test, y_pred_knn)



