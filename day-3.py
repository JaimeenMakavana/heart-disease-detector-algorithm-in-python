from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, matthews_corrcoef, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("heart.csv")
df.fillna(df.select_dtypes(include=['number']).mean(), inplace=True)
df = pd.get_dummies(df, drop_first=True)

num_features = ["age", "sex", "cp", "trestbps"]
num_features = [col for col in num_features if col in df.columns]

if num_features:
    scaler = MinMaxScaler()
    df[num_features] = scaler.fit_transform(df[num_features])

X = df.drop(columns=['target'])
y = df['target']

X_train, y_train, y_test, X_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)

def evaluate_model(y_test, y_pred):
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_pred))
    print("Log Loss:", log_loss(y_test, y_pred))
    print("Matthews Correlation Coefficient:", matthews_corrcoef(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("Logistic Regression Performance:")
evaluate_model(y_test, y_pred_lr)