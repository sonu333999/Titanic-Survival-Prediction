# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv('../data/titanic.csv')

# -------------------- Data Cleaning --------------------

# Fill missing values properly (NO inplace)
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Drop unnecessary column
df = df.drop('Cabin', axis=1)

# Convert categorical to numeric
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Remove any remaining missing values
df = df.dropna()

# -------------------- Feature Selection --------------------
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
y = df['Survived']

# -------------------- Train-Test Split --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------- Model Training --------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -------------------- Prediction --------------------
y_pred = model.predict(X_test)

# -------------------- Evaluation --------------------
print("Accuracy:", accuracy_score(y_test, y_pred))

# -------------------- Save Model --------------------
joblib.dump(model, '../models/titanic_model.pkl')

print("Model saved successfully!")