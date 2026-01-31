import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. Load and Explore the Dataset
# ==========================================
print("Loading dataset...")
try:
    # Reading the CSV file
    df = pd.read_csv('plant_disease_data.csv')
    print("Dataset loaded successfully!")
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    
    print("\nDataset Info:")
    print(df.info())
except FileNotFoundError:
    print("Error: 'plant_disease_data.csv' not found. Please make sure the file exists.")
    exit()

# ==========================================
# 2. Handle Missing Values
# ==========================================
print("\nChecking for missing values...")
print(df.isnull().sum())

# Using SimpleImputer to replace NaN with the mean of the column
# Note: We only impute numerical columns.
print("\nHandling missing values...")
imputer = SimpleImputer(strategy='mean')

# identify feature columns (excluding label)
feature_cols = ['Leaf_Color', 'Texture_Score', 'Spot_Size', 'Moisture_Level']
X = df[feature_cols]
y = df['Disease_Label']

# Fit and transform the features
X = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)

print("Missing values handled.")
print(X.isnull().sum())

# ==========================================
# 3. Split Data into Training and Testing Sets
# ==========================================
print("\nSplitting data into Training (80%) and Testing (20%) sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training Data Size: {X_train.shape}")
print(f"Testing Data Size: {X_test.shape}")

# ==========================================
# 4. Train Decision Tree Classifier
# ==========================================
print("\nTraining Decision Tree Classifier...")
clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X_train, y_train)
print("Model created and trained successfully!")

# ==========================================
# 5. Test the Model and Calculate Accuracy
# ==========================================
print("\nPredicting on test data...")
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Optional: Plot Confusion Matrix if matplotlib/seaborn are available
try:
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=clf.classes_, yticklabels=clf.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    # plt.show() # Uncomment to show plot depending on environment
    print("\nConfusion Matrix plot generated (code commented out to avoid blocking).")
except Exception as e:
    print(f"\nCould not plot confusion matrix: {e}")

# ==========================================
# 6. Predict Disease for New Input
# ==========================================
def predict_new_leaf(color, texture, spot_size, moisture):
    """
    Predicts disease based on new input values.
    Features:
    - Leaf_Color: 0=Green, 1=Yellow, 2=Brown
    - Texture_Score: 1 (Smooth) to 10 (Rough)
    - Spot_Size: size in cm
    - Moisture_Level: percentage (0-100)
    """
    input_data = np.array([[color, texture, spot_size, moisture]])
    prediction = clf.predict(input_data)
    return prediction[0]

print("\n--- Prediction Test with New Data ---")
# Example: Yellow leaf (1), Rough texture (5), Small spot (0.8), Low moisture (35) -> Likely Bacterial Spot
new_leaf_features = [1, 5, 0.8, 35] 
predicted_disease = predict_new_leaf(*new_leaf_features)
print(f"Input Features: {new_leaf_features}")
print(f"Predicted Disease: {predicted_disease}")

# Example: Green leaf (0), Smooth (2), No spot (0.0), Good moisture (65) -> Likely Healthy
new_leaf_features_2 = [0, 2, 0.0, 65]
predicted_disease_2 = predict_new_leaf(*new_leaf_features_2)
print(f"Input Features: {new_leaf_features_2}")
print(f"Predicted Disease: {predicted_disease_2}")

print("\nProject execution completed.")
