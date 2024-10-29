# ASPIRE (Adaptive Scaler and PCA with Intelligent REduction)
*Previously known as AdaptivePCA*

ASPIRE is an enhanced preprocessing and dimensionality reduction system that intelligently adapts to data characteristics through statistical analysis. The model combines adaptive scaling selection with optimized Principal Component Analysis (PCA) to provide an efficient and robust feature reduction solution with minimal computational costs.

ASPIRE represents a significant advancement in automated feature engineering, offering a robust solution for dimensionality reduction while maintaining data integrity and model performance.


## Core Functionality

ASPIRE employs a two-stage adaptive approach:

1. **Intelligent Preprocessing**
   - Comprehensive preprocessing handling; numeric features, missing values, infinity and nan.
   - Performs feature-wise normality testing using Shapiro-Wilk test
   - Automatic selection of the optimal scaler based on data distribution

2. **Dynamic Dimensionality Reduction**
   - Determines the optimal number of PCA components while maintaining a specified variance threshold
   - Early stops to ensure computational efficiency
   - Adapts to dataset dimensions and characteristics
   - Provides comprehensive validation of the reduction effectiveness

## Overall Design Pattern
```bash
Data → Preprocessing → Scaler Selection → PCA Optimization → Validation → Prediction
```

## Key Advantages

- **Automation**: Eliminates manual preprocessing decisions through data-driven selection
- **Adaptivity**: Adjusts preprocessing and reduction strategies based on data characteristics
- **Efficiency**: Optimizes computational resources while maintaining data integrity
- **Validation**: Includes built-in performance comparison framework
- **Transparency**: Provides detailed insights into selection decisions and performance metrics

## Installation

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

## Usage

```python
import pandas as pd
from adaptivepca import AdaptivePCA

# Load your data (example)
data = pd.read_csv("your_dataset.csv")
X = data.drop(columns=['Label'])  # Features
y = data['Label']  # Target variable (Optional)

# Initialize and fit the model to determine the optimal scaler and PCA configuration
adaptive_pca = AdaptivePCA(variance_threshold=0.95, max_components=50, varince_ratio=0.5, normality_ratio=5, verbose=1)
adaptive_pca.fit(X)

# Optional - Validate with a classifier with full and reduced dataset performance
classifier = adaptive_pca.validate_with_classifier(X, y, classifier=DecisionTreeClassifier, test_size=0.2, cv=5)

# Optional - Run prediction with classifier, show output of confusion matrix, classification report, inference time, fpr, far, specificity, auc-roc, mcc
adaptive_pca.predict_with_classifier(X, y)

# Optional - View model configurations
adaptive_pca.view_config()

# Optional - Export the model in joblib format
adaptive_pca.export_model("your_model_name.joblib", classifier)

```

# Key Components

## Initialization Parameters
- `variance_threshold`: Target explained variance (default: 0.95)
- `max_components`: Maximum PCA components to consider (default: 50)
- `variance_ratio`: Variance ratio threshold (default: 5.0)
- `normality_ratio`: P-value threshold for Shapiro-Wilk test (default: 0.05)
- `verbose`: Logging detail level (default: 0)

## Methods
- `fit(X)`: Fits the AdaptivePCA model to the data `X`.
- `transform(X)`: Transforms the data `X` using the fitted PCA model.
~~- `fit_transform(X)`: Fits and transforms the data in one step.~~
- `validate_with_classifier(X, y, classifier=None, cv=5, test_size=0.2)`: Tests model performance.
- `predict_with_classifier(X, y)`: Makes predictions using trained classifier.
- `export_model(model_name, classifier)`: Saves model to file.
- `view_config()`: Shows current configuration.

## Main Algorithms Flow

### 1. Data Preprocessing
```bash
Input: DataFrame X
Output: Clean DataFrame

# Clean the data
Keep only numeric columns
For each column:
    Replace infinities with max/min values
    Fill missing values with column mean
Remove constant columns (variance = 0)
```

### 2. Scaler Selection
```bash
Input: DataFrame X
Output: Best scaler for the data

For each column:
    Take sample of up to 5000 points
    Test if data is normal (Shapiro-Wilk test)
    Count if normal or not normal

If more normal columns:
    Return StandardScaler
Else:
    Return MinMaxScaler
```

### 3. PCA Optimal Component Selection
```bash
Input: Scaled data X
Output: Optimal number of components

Set max_components = min(50, number of features)
Try components from 1 to max_components:
    Calculate explained variance
    If variance >= threshold (default 95%):
        Return current number of components
```

### 4. Main Fitting Process
```bash
Input: DataFrame X
Output: Fitted model

Clean the data
Choose and fit best scaler
Scale the data
Find best number of components
Save configuratio
```

### 4. Transform Data
```bash
Input: New DataFrame X
Output: Reduced data

Clean the data using saved settings
Scale data using saved scaler
Reduce dimensions using saved components
Return reduced data
```

### 5. Validation
```bash
Input: Data X, Labels y
Output: Performance metrics

# Compare original vs reduced data
Train model on original data
Train model on reduced data
Compare:
    - Accuracy
    - Speed
    - Memory usage
```

### 6. Prediction
```bash
Input: New data X
Output: Predictions

Clean the data
Apply scaling
Reduce dimensions
Make predictions
Return results and metrics
```

### 7. Save Model
```bash
Input: Model name, Trained model
Output: Saved file

Collect:
    - Parameters
    - Scaler
    - Components
    - Column names
Save everything to file
```

## Key Features

### Adaptivity Mechanisms
1. **Scaler Selection**:
   - Based on feature-wise normality tests
   - Considers data distribution characteristics
   - Defaults sensibly for edge cases

2. **Component Selection**:
   - Adapts to data dimensions
   - Respects variance threshold
   - Limits maximum components

3. **Validation**:
   - Supports both cross-validation and train-test split
   - Compares performance with original data
   - Measures computational efficiency gains

### Error Handling
- Handles zero-variance features
- Manages missing values through mean imputation
- Validates presence of numeric columns
- Ensures fit before transform

## Complexity Analysis
- Time Complexity: O(n * d^2) for PCA computation
- Space Complexity: O(n * d) for data storage
Where n = samples, d = features

## Use Cases
ASPIRE is particularly valuable for:
- Machine learning pipelines requiring automated preprocessing
- High-dimensional data analysis
- Feature engineering optimization
- Model performance enhancement
- Exploratory data analysis

## Technical Foundation
The system integrates:
- Statistical testing for data distribution analysis
- Adaptive scaling techniques
- Principal Component Analysis
- Machine learning validation frameworks
- Performance optimization methods

## Performance Comparison: AdaptivePCA vs. Traditional PCA Optimization (GridSearch)

### Speed

AdaptivePCA leverages parallel processing to evaluate scaling and PCA component selection concurrently. In our tests, AdaptivePCA achieved up to a 95% reduction in processing time compared to the traditional PCA method. This is especially useful when working with high-dimensional data, where traditional methods may take significantly longer due to sequential grid search.

### Explained Variance

Both AdaptivePCA and traditional PCA achieve similar levels of explained variance, with AdaptivePCA dynamically selecting the number of components based on a defined variance threshold. Traditional PCA, on the other hand, requires manual parameter tuning, which can be time-consuming.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request to discuss your changes.

## Acknowledgments
This project makes use of the `scikit-learn`, `numpy`, and `pandas` libraries for data processing and machine learning.

## Version Update Log
- `1.0.3` - Added flexibility in scaling, fix error handling when max_components exceeding the available number of features or samples.
- `1.0.6` - Added Parameter verbose as an argument to __init__, with a default value of 0.
- `1.1.0` - Added validation, prediction with classifier, clean up the code.
