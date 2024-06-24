
---

# Liver Disease Prediction Model

This repository contains a Jupyter Notebook for building and evaluating a machine learning model to predict liver disease based on patient records. The model uses multiple classifiers and combines their predictions to provide a final diagnosis.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Data Preprocessing](#data-preprocessing)
- [Model Building](#model-building)
- [Model Evaluation](#model-evaluation)
- [Saving the Model](#saving-the-model)
- [Contributing](#contributing)
- [License](#license)

## Overview

The notebook builds and evaluates several machine learning models to predict liver disease. The steps include:

1. Loading and exploring the dataset.
2. Data preprocessing.
3. Splitting the data into training and testing sets.
4. Building multiple classifiers.
5. Evaluating the models using various metrics.
6. Saving the best model.

## Dataset

The dataset used in this notebook consists of medical attributes related to liver disease. Each row represents a patient with various attributes and the diagnosis result. The dataset can be found [here](https://www.kaggle.com/datasets/uciml/indian-liver-patient-records).

### Dataset Description

The dataset contains the following columns:

- **Age**: Age of the patient
- **Gender**: Gender of the patient
- **Total_Bilirubin**: Total Bilirubin
- **Direct_Bilirubin**: Direct Bilirubin
- **Alkaline_Phosphotase**: Alkaline Phosphotase
- **Alamine_Aminotransferase**: Alamine Aminotransferase
- **Aspartate_Aminotransferase**: Aspartate Aminotransferase
- **Total_Protiens**: Total Proteins
- **Albumin**: Albumin
- **Albumin_and_Globulin_Ratio**: Albumin and Globulin Ratio
- **Dataset**: Field used to split the data into training (1) and testing (2)

## Installation

To run this notebook, you need to have Python installed along with the following packages:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

You can install these packages using pip:

```sh
pip install pandas numpy matplotlib seaborn scikit-learn
```

Alternatively, you can use the provided `requirements.txt` file:

```sh
pip install -r requirements.txt
```

## Usage

1. Clone this repository:

```sh
git clone [<repository-url>](https://github.com/Sahiru2007/Liver-Disease-Prediction-Model.git)
cd Liver-Disease-Prediction-Model
```

2. Open the Jupyter Notebook:

```sh
jupyter notebook liver_disease.ipynb
```

3. Run all cells in the notebook to see the complete analysis and model evaluation.

## Data Preprocessing

### Handling Missing Values

Missing values in the dataset are handled by replacing them with the median of the respective columns.

### Encoding Categorical Variables

The gender column is encoded into numeric values using label encoding.

```python
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])
```

### Splitting the Data

The dataset is split into training and testing sets based on the 'Dataset' column:

```python
train_data = data[data['Dataset'] == 1]
test_data = data[data['Dataset'] == 2]

X_train = train_data.drop(['Dataset'], axis=1)
y_train = train_data['Dataset']

X_test = test_data.drop(['Dataset'], axis=1)
y_test = test_data['Dataset']
```

## Model Building

### Models Used

The notebook evaluates several models:

- **Logistic Regression**
- **Random Forest**
- **Support Vector Machine (SVM)**

### Training the Models

Example: Training a Logistic Regression Classifier

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## Model Evaluation

### Metrics

The models are evaluated using the following metrics:

- **Accuracy**: The ratio of correctly predicted instances to the total instances.
- **Precision**: The ratio of true positive instances to the total predicted positives.
- **Recall**: The ratio of true positive instances to the actual positives.
- **F1 Score**: The harmonic mean of precision and recall.
- **Confusion Matrix**: A summary of prediction results on a classification problem.

### Example: Evaluating Logistic Regression Classifier

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'Confusion Matrix: \n{cm}')
```

### Evaluation Results

- **Logistic Regression**: Accuracy ~ 72%
- **Random Forest**: Accuracy ~ 78%
- **SVM**: Accuracy ~ 75%

## Unique Aspects of the Notebook

- **Correlation Matrix**: A heatmap to visualize the correlations between features and the target variable.
- **ROC Curve**: Receiver Operating Characteristic curve to evaluate the trade-off between sensitivity and specificity.

### Correlation Matrix

```python
import seaborn as sns
import matplotlib.pyplot as plt

corr = data.corr()
plt.figure(figsize=(12,8))
sns.heatmap(corr, annot=True, fmt='.2f')
plt.title('Correlation Matrix')
plt.show()
```

### ROC Curve

```python
from sklearn.metrics import roc_curve, auc

y_pred_prob = model.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
```

## Saving the Model

The best-performing model is saved using the `pickle` module for future use:

```python
import pickle

filename = 'liver_disease_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(model, file)

print(f"Model saved to {filename}")
```

## Contributing

Contributions are welcome! Please fork this repository and submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

---
