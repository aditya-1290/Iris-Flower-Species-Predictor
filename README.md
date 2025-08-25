# Iris Flower Species Predictor

A machine learning project that classifies Iris flowers into three species (Setosa, Versicolor, Virginica) based on their sepal and petal measurements. This project demonstrates industry best practices for classification tasks including data preprocessing, model evaluation, hyperparameter tuning, and deployment.

## Features

- **Multiple Algorithms**: Implements Logistic Regression, K-Nearest Neighbors, Support Vector Machines, Decision Trees, and Random Forest
- **Comprehensive Evaluation**: Includes accuracy scores, cross-validation, confusion matrix, and classification reports
- **Hyperparameter Tuning**: Uses GridSearchCV to find optimal parameters
- **Interactive Web App**: Streamlit-based interface for real-time predictions
- **Visualizations**: EDA plots, feature importance, and model comparison charts

## Dataset

The project uses the classic Iris dataset from `sklearn.datasets.load_iris()` which contains:
- 150 samples (50 per class)
- 4 features: sepal length, sepal width, petal length, petal width
- 3 target classes: Setosa, Versicolor, Virginica

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Iris_Flower_Species_Predictor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Train the Model
Run the model training script to train and evaluate multiple algorithms:
```bash
python iris_model.py
```

This will:
- Perform exploratory data analysis
- Train and evaluate multiple classification algorithms
- Perform hyperparameter tuning on the best model
- Save the best model as `best_model.pkl`
- Generate visualizations (EDA, confusion matrix, feature importance, model comparison)

### 2. Run the Web App
Start the Streamlit web application:
```bash
streamlit run app.py
```

The app will open in your browser where you can:
- Adjust sliders for sepal and petal measurements
- Get instant predictions of the Iris species
- See prediction probabilities for each class

## Project Structure

```
Iris_Flower_Species_Predictor/
├── iris_model.py          # Main model training and evaluation script
├── app.py                 # Streamlit web application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── best_model.pkl        # Saved best model (generated after training)
├── iris_eda.png          # EDA visualizations (generated)
├── confusion_matrix.png  # Confusion matrix (generated)
├── feature_importance.png # Feature importance plot (generated)
└── model_comparison.png  # Model comparison chart (generated)
```

## Model Performance

The project evaluates multiple algorithms and selects the best performing one. Typical results:

| Model | Accuracy | Cross-Validation Score |
|-------|----------|-----------------------|
| Random Forest | ~0.97 | ~0.96 ± 0.02 |
| SVM | ~0.97 | ~0.96 ± 0.03 |
| Logistic Regression | ~0.97 | ~0.95 ± 0.04 |
| K-Nearest Neighbors | ~0.97 | ~0.96 ± 0.03 |
| Decision Tree | ~0.93 | ~0.92 ± 0.04 |

## Key Concepts Demonstrated

- **Data Preprocessing**: Feature scaling, train-test splitting
- **Model Evaluation**: Accuracy, confusion matrix, classification report
- **Cross-Validation**: 5-fold cross-validation for robust evaluation
- **Hyperparameter Tuning**: Grid search for optimal parameters
- **Feature Importance**: Understanding which features drive predictions
- **Model Deployment**: Web interface for real-time predictions

## Technologies Used

- Python 3.8+
- scikit-learn
- pandas
- matplotlib
- seaborn
- streamlit
- numpy

## Contributing

Feel free to contribute by:
- Adding new classification algorithms
- Improving the web interface
- Adding more visualizations
- Implementing additional evaluation metrics

## License

This project is open source and available under the MIT License.
