import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

X_train = train.drop(columns=["is_fraud"])
y_train = train["is_fraud"]
X_test = test.drop(columns=["is_fraud"])
y_test = test["is_fraud"]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)
log_preds = log_model.predict(X_test_scaled)

tree_model = DecisionTreeClassifier()
tree_model.fit(X_train_scaled, y_train)
tree_preds = tree_model.predict(X_test_scaled)

forest_model = RandomForestClassifier(n_estimators=100)
forest_model.fit(X_train_scaled, y_train)
forest_preds = forest_model.predict(X_test_scaled)

print("Logistic Regression Results")
print(classification_report(y_test, log_preds))
print(confusion_matrix(y_test, log_preds))

print("Decision Tree Results")
print(classification_report(y_test, tree_preds))
print(confusion_matrix(y_test, tree_preds))

print("Random Forest Results")
print(classification_report(y_test, forest_preds))
print(confusion_matrix(y_test, forest_preds))

joblib.dump(forest_model, "fraud_model.joblib")
joblib.dump(scaler, "scaler.joblib")
