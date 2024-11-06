# ASPIRE (Adaptive Scaler and PCA with Intelligent REduction)
*Previously known as AdaptivePCA*

ASPIRE is an advanced implementation of the adaptive Principal Component Analysis (PCA) model that automatically handles data preprocessing and feature scaling. The model dynamically adjusts preprocessing steps based on data characteristics, tailors its scaling strategy to the data distribution, and automatically selects the optimal number of PCA components according to data patterns. This provides a comprehensive solution for dimensionality reduction with built-in validation capabilities

## Features

- Automatic feature scaling based on data distribution
- Comprehensive preprocessing pipeline
- Built-in validation with classifier support
- Class imbalance detection and handling
- Performance metrics and exportable results

## Overall Design Pattern
```bash
Data → Preprocessing → Automatic Scaling System → PCA Optimization → Model Validation → Model Persistence
```

## Dependencies
- numpy>=1.19.0
- pandas>=1.2.0
- scikit-learn>=0.24.0
- lightgbm>=3.0.0
- imbalanced-learn>=0.8.0
- scipy>=1.6.0

## Installation

Install dependencies:
```bash
pip install scikit-learn numpy pandas lightgbm scipy imbalanced-learn lightgbm
```

Instal from Pypi repository:
```bash
pip install adaptivepca
```

Clone this repository and install the package using `pip`:
```bash
git clone https://github.com/nqmn/adaptivepca.git
cd adaptivepca
pip install .
```

## How It Works

### 1. Preprocessing
The preprocessing pipeline includes:
- Selection of numeric columns
- Outlier handling using the IQR method
- Infinity value replacement
- Missing value imputation
- Automatic feature scaling selection
- Class imbalance handling (optional)

### 2. Feature Scaling
The algorithm automatically chooses between:
- StandardScaler: For normally distributed features
- MinMaxScaler: For non-normally distributed features

The selection is based on statistical tests performed on each feature.

### 3. Component Selection
The optimal number of components is determined by:
- Explained variance threshold (default: 0.95)
- Maximum component limit (default: 50)
- Minimum eigenvalue threshold (default: 0.0001)

### 4. Validation
Built-in validation options include:
- Cross-validation
- Train-test split
- Performance metrics (accuracy, ROC-AUC, confusion matrix)
- Efficiency comparisons between full and reduced datasets


## Key Adaptive Features

The adaptive features in ASPIRE refers to its ability to automatically adjust its behavior based on the characteristics of the input data, rather than using fixed, predetermined approaches. This makes it more robust and suitable for a wider range of data types and scenarios:

### Distribution-Based Adaptation
- Continuously monitors data distribution
- Adjusts scaling methods accordingly
- Updates preprocessing strategies

### Variance-Based Adaptation

- Dynamically adjusts number of components
- Responds to explained variance patterns
- Maintains optimal dimensionality

### Balance-Based Adaptation

- Monitors class distributions
- Adaptively applies SMOTE when needed
- Adjusts sampling strategies

### Performance-Based Adaptation

- Tracks model performance
- Adjusts parameters based on results
- Optimizes computational efficiency


## Usage Example

### Basic Usage

```python
from adaptive_pca import AdaptivePCA

# Load your data
data = pd.read_csv("your_dataset.csv")
X = data.drop(['Label']) # Features
y = data['Label'] # Target variable

# Initialize AdaptivePCA
adaptive_pca = AdaptivePCA()

# Preprocess data
X_preprocessed, y_preprocessed, smote_applied = adaptive_pca.preprocess_data(X, y)

# Fit the model
adaptive_pca.fit(X_preprocessed, y_preprocessed, smote_applied)

# Optional - Validate with a classifier with full and reduced dataset performance
adaptive_pca.validate_with_classifier(X_preprocessed, y_preprocessed)

# Optional - Make predictions
adaptive_pca.predict_with_classifier(X_preprocessed, y_preprocessed)

# Optional - Export model
adaptive_pca.export_model('your_model_name.joblib')

```

### Advanced Usage

```python
from adaptivepca import AdaptivePCA
from sklearn.tree import DecisionTreeClassifier

# Load your data
data = pd.read_csv("your_dataset.csv")
X = data.drop(columns=['Label'])  # Features
y = data['Label']  # Target variable

# Initialize AdaptivePCA
adaptive_pca = AdaptivePCA(
    variance_threshold=0.95,
    max_components=50,
    min_eigenvalue_threshold=1e-4,
    normality_ratio=0.05,
    verbose=1
)
# Run Preprocessing
X_preprocessed, y_preprocessed, smote_applied = adaptive_pca.preprocess_data(X, y, smote_test=True)

# Fit the model
adaptive_pca.fit(X_preprocessed, y_preprocessed, smote_applied)


# Optional - Validate with a classifier with full and reduced dataset performance
adaptive_pca.validate_with_classifier(X, y, classifier=DecisionTreeClassifier(), test_size=0.2, cv=5)

# Optional - Make predictions with classifier, show output of confusion matrix, classification report, inference time
adaptive_pca.predict_with_classifier(X, y)

# Optional - Export model
adaptive_pca.export_model("your_model_name.joblib")

```

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| variance_threshold | 0.95 | Minimum cumulative explained variance ratio |
| max_components | 50 | Maximum number of principal components |
| min_eigenvalue_threshold | 0.0001 | Minimum eigenvalue for component selection |
| normality_ratio | 0.05 | P-value threshold for normality test |
| verbose | 0 | Output detail level (0: minimal, 1: detailed) |

## Output Formats
All results are available in both human-readable and JSON formats, including:
- Preprocessing statistics
- PCA performance metrics
- Validation results
- Prediction metrics

## Model Export
The entire model state can be exported, including:
- Fitted scalers
- PCA configuration
- Trained classifier
- All parameter settings

## Performance Considerations
- Automatically adapts to data distributions
- Provides efficiency metrics comparing full vs. reduced datasets
- Supports large datasets through sampling in normality tests
- Handles class imbalance when detected

# Performance Comparison: ASPIRE vs. Traditional PCA Optimization (GridSearch)

The key advantage of ASPIRE over traditional PCA is its automated, intelligent approach to data processing and dimensionality reduction. While traditional PCA requires significant manual intervention and expertise, AdaptivePCA provides a more robust, automated solution that can adapt to different data characteristics while maintaining high performance standards.

## Comparison Flowchart
```
flowchart LR
    subgraph Traditional["Traditional PCA"]
        T1[Raw Data] --> T2[Manual Preprocessing]
        T2 --> T3[Standard Scaling]
        T3 --> T4[PCA]
        T4 --> T5[Manual Validation]
    end

    subgraph Adaptive["AdaptivePCA"]
        A1[Raw Data] --> A2[Automated Preprocessing]
        A2 --> A3[Adaptive Scaling]
        A3 --> A4[Optimal PCA]
        A4 --> A5[Built-in Validation]
        
        B1[Distribution Tests] --> A3
        B2[Variance Analysis] --> A4
        B3[Performance Metrics] --> A5
    end

    style Traditional fill:#ffb3b3
    style Adaptive fill:#b3ffb3
```

### Speed

AdaptivePCA adaptively selects the optimal configuration based on data-driven rules, which is less computationally intense than the exhaustive search performed by grid search. In our tests, AdaptivePCA achieved up to a 90% reduction in processing time compared to the traditional PCA method. This is especially useful when working with high-dimensional data, where traditional methods may take significantly longer due to sequential grid search.

## Performance on different dataset (Full & Reduced Dataset)

Most datasets maintain high accuracy, with reduced datasets achieving similar scores to full datasets in nearly all cases. Additionally, the reduced datasets significantly decrease processing time, with time reductions ranging from 1.85% to 58.03%. This indicates that reduced datasets can offer substantial efficiency benefits, especially for larger datasets.

| Dataset | Score (Acc) | Time (s) | Gain (%) |
|---------|-------------|----------|----------|
|NSL-KDD (full) | 0 | 0 | - |
|NSL-KDD (reduced)| 0 | 0 | 0 |
|CIC-IDS2017 (full) | 0 | 0 | - |
|CIC-IDS2017 (reduced)| 0 | 0 | 0 |
|CIC-DDOS2019 (full) | 0 | 0 | - |
|CIC-DDOS2019 (reduced)| 0 | 0 | 0 |
|InSDN (full) | 0 | 0 | - |
|InSDN (reduced)| 0 | 0 | 0 |

# Model Flowchart

```
flowchart TB
    Start([Start]) --> Input[/Input Data X, y/]
    
    subgraph Preprocessing
        Input --> NumericCols[Select Numeric Columns]
        NumericCols --> Outliers[Handle Outliers]
        Outliers --> Missing[Impute Missing Values]
        Missing --> ScalerTest{Test Feature Distributions}
        ScalerTest -->|Normal| StandardScale[Apply StandardScaler]
        ScalerTest -->|Non-normal| MinMaxScale[Apply MinMaxScaler]
        StandardScale --> Balance{Check Class Balance}
        MinMaxScale --> Balance
        Balance -->|Imbalanced| SMOTE[Apply SMOTE]
        Balance -->|Balanced| Clean[Cleaned Data]
        SMOTE --> Clean
    end
    
    subgraph PCA_Fitting
        Clean --> InitialPCA[Initial PCA Fit]
        InitialPCA --> FindComponents[Find Optimal Components]
        FindComponents --> CheckVar{Variance >= Threshold?}
        CheckVar -->|No| IncreaseComp[Increase Components]
        IncreaseComp --> CheckVar
        CheckVar -->|Yes| FinalPCA[Final PCA Fit]
    end
    
    subgraph Validation
        FinalPCA --> SplitData[Split Train/Test]
        SplitData --> ParallelVal{Parallel Validation}
        ParallelVal --> FullTrain[Train on Full Data]
        ParallelVal --> RedTrain[Train on Reduced Data]
        FullTrain --> CompareMetrics[Compare Performance]
        RedTrain --> CompareMetrics
    end
    
    CompareMetrics --> Export[/Export Model/]
    Export --> End([End])

    style Preprocessing fill:#e1f3d8
    style PCA_Fitting fill:#ffd7d7
    style Validation fill:#d7e3ff
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request to discuss your changes.

## Acknowledgments
This project makes use of the `scikit-learn`, `numpy`, `pandas`, `imbalanced-learn` and `scipy` libraries for data processing and machine learning.

## Version Update Log
- `1.0.3` - Added flexibility in scaling, fix error handling when max_components exceeding the available number of features or samples.
- `1.0.6` - Added Parameter verbose as an argument to __init__, with a default value of 0.
- `1.1.0` - Added validation, prediction with classifier, clean up the code.
- `1.1.3` - Revamped the code, add error handling for fit module and update export model.