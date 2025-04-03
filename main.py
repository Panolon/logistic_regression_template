# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            precision_recall_curve, roc_curve)
from sklearn.pipeline import Pipeline


# import data .csv
file_path = ""
df = pd.read_csv(file_path, sep=',', encoding='utf-8', decimal='.')
features = df.copy()
features.head()
features.info()
features.describe()

# Data preprocessing

# Get a list of all datetime columns
date_columns = df.select_dtypes(include=['datetime64', 'datetime']).columns
features = features.drop(columns=date_columns) # remove datetime columns
#features.head()

# Check for missing values

#features = features.dropna() # remove rows with missing values
features = features.dropna(axis=1) # remove columns with missing values
#features = features.fillna(features.mean()) # fill missing values with mean or median/mode/custom_value

# select index column
#features.set_index(index_col[0], inplace=True)
#features.head()

# Convert categorical variables to numerical with label encoding -- additional one-hot encoding
categorical_cols = features.select_dtypes(include=['object']).columns
for col in categorical_cols:
    le = LabelEncoder()
    features[col] = le.fit_transform(features[col])

# Choose target column and feature columns
target_col = 'HeartDisease'
features = features.drop(columns=[target_col]) 
target = df[target_col]

# Standardize the data
scaler = StandardScaler()
col_names = features.columns
features = scaler.fit_transform(features)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=42, stratify=target)

# create classifier and hyperparameter tuning
model = LogisticRegression(max_iter=10000, solver='saga', class_weight='balanced')
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2', 'elasticnet'],
    'l1_ratio': [0.0, 0.5, 1.0], # only used for elasticnet penalty
}

# Grid search with 5-fold CV
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_
print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

print(pd.DataFrame(grid_search.cv_results_).sort_values(by='rank_test_score').head(10))

# Model evaluation
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

# Metrics
metrics = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1 Score': f1_score(y_test, y_pred),
    'ROC-AUC': roc_auc_score(y_test, y_proba)
}

print("\nEvaluation Metrics:")
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# 4. Feature Coefficients
coefficients = best_model.coef_[0]
attributes = col_names
    
# Calculate log odds
log_odds = pd.DataFrame({
    'Feature': attributes,
    'Coefficient': coefficients,
    'Odds': np.exp(coefficients),
    'Probabilities': np.exp(coefficients) / (1 + np.exp(coefficients))
})

print("\nLog Odds Analysis:")
print(log_odds.sort_values('Coefficient', ascending=False))

# 5. Pearson Correlation Heatmap
print("\nPearson Correlation Heatmap:")
correlation_matrix = pd.DataFrame(X_train, columns=col_names).corr()

# Visualization

plt.figure(figsize=(18, 12))

# 1. Probability Scatter Plot
plt.subplot(2, 3, 1)
plt.scatter(range(len(y_proba)), y_proba, c=y_test, cmap='coolwarm', alpha=0.6)
plt.colorbar(label='True Class (0/1)')
plt.title('Predicted Probabilities by Instance')
plt.xlabel('Instance Index')
plt.ylabel('Predicted Probability')
plt.grid(True)
# Add a vertical line at the threshold of 0.5
plt.axhline(y=0.5, color='black', linestyle='--', label='Threshold = 0.5')
plt.legend()

# 2. Precision-Recall Curve
precision, recall, thresholds_pr = precision_recall_curve(y_test, y_proba)
plt.subplot(2, 3, 2)
plt.plot(recall, precision, label='PR Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall Curve (AP={metrics["Precision"]:.2f})')
plt.fill_between(recall, precision, alpha=0.1)
plt.grid(True)
plt.legend()

# 3. ROC Curve
fpr, tpr, thresholds_roc = roc_curve(y_test, y_proba)
plt.subplot(2, 3, 3)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {metrics["ROC-AUC"]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.fill_between(fpr, tpr, alpha=0.1)
plt.grid(True)
plt.legend()

# 4. Threshold Analysis
plt.subplot(2, 3, 4)
plt.plot(thresholds_pr, precision[:-1], label='Precision')
plt.plot(thresholds_pr, recall[:-1], label='Recall')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision/Recall vs Threshold')
plt.legend()
plt.grid(True)
plt.tight_layout()

# 5. Heatmap
plt.subplot(2, 3, 5)
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
plt.title("Pearson Correlation Heatmap")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

# 6. Feature Coefficients Bar Plot
plt.subplot(2, 3, 6)
plt.barh(attributes, coefficients)
plt.title('Feature Coefficients')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.grid(True)

plt.show()

