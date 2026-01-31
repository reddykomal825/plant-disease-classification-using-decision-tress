# Plant Disease Detection Using Decision Tree

## 🌱 Project Overview
This is a beginner-friendly machine learning project that detects whether a plant leaf is healthy or diseased using the **Decision Tree Algorithm**. It analyzes leaf features like color, texture, spot size, and moisture to predict specific diseases.

---

## 📂 Project Structure
- `plant_disease_detection.py`: The main Python source code.
- `plant_disease_data.csv`: The dataset used for training and testing.
- `requirements.txt`: List of required Python libraries.
- `README.md`: Project documentation (this file).

---

## ⚙️ How to Run
1. **Install Dependencies**:
   Open your terminal or command prompt and run:
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the Script**:
   ```bash
   python plant_disease_detection.py
   ```
3. **Output**:
   The script will display the dataset info, training accuracy, confusion matrix, and prediction results for new sample leaves.

---

## 📊 Flow Diagram

```mermaid
graph TD;
    A[Start] --> B[Load Dataset];
    B --> C[Data Preprocessing];
    C --> D[Handle Missing Values];
    D --> E[Split Data (Train/Test)];
    E --> F[Train Decision Tree Model];
    F --> G[Evaluate Model Accuracy];
    G --> H[Predict New Leaf Disease];
    H --> I[Display Result];
    I --> J[End];
```

---

## 📝 Code Explanation

1. **Import Libraries**: `pandas` for data handling, `sklearn` for machine learning, `matplotlib`/`seaborn` for visualization.
2. **Load Data**: Reads `plant_disease_data.csv` into a dataframe.
3. **Preprocessing**:
   - Checks for missing values (`NaN`).
   - Uses `SimpleImputer` to fill missing values with the mean.
4. **Train/Test Split**: Splits data into 80% training and 20% testing to evaluate performance fairly.
5. **Decision Tree Classifier**:
   - `DecisionTreeClassifier(criterion='entropy')` is initialized.
   - Fits the model on training data.
6. **Evaluation**:
   - Calculates **Accuracy**.
   - Generates a **Confusion Matrix** to show correct vs. incorrect predictions.
7. **Prediction**: A custom function `predict_new_leaf()` takes inputs (color, texture, etc.) and outputs the predicted disease.

---

## 📈 Advantages & Disadvantages

### Advantages
- **Simple to Understand**: Decision Trees mimic human decision-making.
- **Requires Little Data Preparation**: Can handle numerical and categorical data (with some encoding).
- **Interpretability**: easy to visualize the tree structure.

### Disadvantages
- **Overfitting**: Can create very complex trees that don't generalize well if not pruned.
- **Instability**: Small variations in data might result in a completely different tree.

---

## 🚀 Future Scope
- **Image Processing**: Use Convolutional Neural Networks (CNN) to detect diseases directly from leaf images.
- **Mobile App**: Integrate the model into a mobile app for farmers.
- **More Diseases**: Expand the dataset to cover more crop types and diseases.

---

## 🎓 Conclusion
This project demonstrates how Machine Learning can be applied to agriculture. By using a Decision Tree, we created a simple yet effective system to classify plant health based on visible features.
