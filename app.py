# Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# Load Dataset
df = pd.read_csv('./cardio_train.csv', delimiter=';')

# Data Pre-processing
# Handle missing values
df.dropna(inplace=True)

# Encode categorical variables if any (e.g., gender)
df['gender'] = df['gender'].map({1: 0, 2: 1})  # Assuming 1: Female, 2: Male

# Separate features and target variable
# Safely drop 'id' and 'cardio' columns if they exist
columns_to_drop = ['id', 'cardio']
X = df.drop(columns=columns_to_drop, errors='ignore')
y = df['cardio']  # Target column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature variables
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Data Analysis and Visualizations
# Visualize Distributions
sns.histplot(df['age'], kde=True)
plt.title('Age Distribution')
plt.show()

sns.boxplot(x=df['age'])
plt.title('Age Boxplot')
plt.show()

# Visualize the target variable distribution
sns.countplot(x=y)
plt.title('Target Variable Distribution')
plt.show()

# Correlation Matrix
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Machine Learning Models
models = {
    'Support Vector Machine': SVC(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier()
}

# Train and evaluate each model
results = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[model_name] = accuracy
    print(f"Model: {model_name}")
    print("Accuracy:", accuracy)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\n")

# Build the Best Model
# Select the best model based on accuracy
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

# Retrain the best model on the entire dataset
best_model.fit(X, y)

print(f"The best model is {best_model_name} with an accuracy of {results[best_model_name]}")

# Save the model for future use
joblib.dump(best_model, 'best_cardiovascular_disease_model.pkl')

# Example usage of the saved model
# loaded_model = joblib.load('best_cardiovascular_disease_model.pkl')
# predictions = loaded_model.predict(new_data)
