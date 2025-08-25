import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print("Iris Dataset Information:")
print(f"Features: {feature_names}")
print(f"Target classes: {target_names}")
print(f"Dataset shape: {X.shape}")
print(f"Class distribution: {np.bincount(y)}")

# Create DataFrame for better visualization
df = pd.DataFrame(X, columns=feature_names)
df['species'] = y
df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("\nDataset Sample:")
print(df.head())

# Exploratory Data Analysis
print("\n=== Exploratory Data Analysis ===")
print("\nBasic Statistics:")
print(df.describe())

print("\nCorrelation Matrix:")
correlation_matrix = df[feature_names].corr()
print(correlation_matrix)

# Visualizations
plt.figure(figsize=(15, 10))

# Pairplot
plt.subplot(2, 2, 1)
sns.pairplot(df, hue='species_name', palette='viridis')
plt.title('Pairplot of Iris Features')

# Correlation heatmap
plt.subplot(2, 2, 2)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap')

# Box plots for each feature
plt.subplot(2, 2, 3)
df_melted = df.melt(id_vars=['species_name'], value_vars=feature_names)
sns.boxplot(data=df_melted, x='variable', y='value', hue='species_name')
plt.title('Feature Distribution by Species')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('iris_eda.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nEDA visualizations saved as 'iris_eda.png'")

# Data Preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Model Training and Evaluation
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Support Vector Machine': SVC(probability=True, random_state=42),  # Enable probability estimates
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42)
}

results = {}

print("\n=== Model Evaluation ===")
for name, model in models.items():
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
    print(f"\n{name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Cross-validation: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Detailed classification report for the best model
    if accuracy == max([results[m]['accuracy'] for m in results]):
        print(f"\nDetailed Classification Report for {name}:")
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=target_names, yticklabels=target_names)
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Confusion matrix saved as 'confusion_matrix.png'")

# Hyperparameter Tuning for the best model
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
print(f"\n=== Hyperparameter Tuning for {best_model_name} ===")

if best_model_name == 'Logistic Regression':
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs']
    }
elif best_model_name == 'K-Nearest Neighbors':
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance']
    }
elif best_model_name == 'Support Vector Machine':
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 0.01],
        'kernel': ['rbf', 'linear']
    }
elif best_model_name == 'Decision Tree':
    param_grid = {
        'max_depth': [None, 3, 5, 7, 10],
        'min_samples_split': [2, 5, 10]
    }
else:  # Random Forest
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10]
    }

grid_search = GridSearchCV(
    models[best_model_name], 
    param_grid, 
    cv=5, 
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train_scaled, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Train final model with best parameters
best_model = grid_search.best_estimator_
final_accuracy = accuracy_score(y_test, best_model.predict(X_test_scaled))
print(f"Final test accuracy: {final_accuracy:.4f}")

# Feature Importance (for tree-based models)
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='importance', y='feature', palette='viridis')
    plt.title('Feature Importance')
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Feature importance plot saved as 'feature_importance.png'")

# Save the best model and scaler
with open('best_model.pkl', 'wb') as f:
    pickle.dump({
        'model': best_model,
        'scaler': scaler,
        'feature_names': feature_names,
        'target_names': target_names
    }, f)

print(f"\nBest model saved as 'best_model.pkl'")
print(f"Final model: {best_model_name}")
print(f"Final accuracy: {final_accuracy:.4f}")

# Model comparison
print("\n=== Model Comparison ===")
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['accuracy'] for m in results],
    'CV Mean': [results[m]['cv_mean'] for m in results],
    'CV Std': [results[m]['cv_std'] for m in results]
}).sort_values('Accuracy', ascending=False)

print(comparison_df)

plt.figure(figsize=(12, 6))
sns.barplot(data=comparison_df, x='Accuracy', y='Model', palette='viridis')
plt.title('Model Accuracy Comparison')
plt.xlim(0, 1)
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("Model comparison plot saved as 'model_comparison.png'")
