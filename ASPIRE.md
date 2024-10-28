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

# Initialize and fit AdaptivePCA
# Make sure to use cleaned dataset. Eg. remove missing, etc.
adaptive_pca = AdaptivePCA(variance_threshold=0.95, max_components=50, scaler_test=True, verbose=1)
X_reduced = adaptive_pca.fit_transform(X)
```

# AdaptivePCA Algorithm

## Initialization Parameters
- `variance_threshold`: Target explained variance (default: 0.95)
- `max_components`: Maximum PCA components to consider (default: 50)
- `variance_ratio`: Variance ratio threshold (default: 5.0)
- `normality_ratio`: P-value threshold for Shapiro-Wilk test (default: 0.05)
- `verbose`: Logging detail level (default: 0)

## Main Algorithm Flow

### 1. Data Preprocessing (fit method)
```
Input: DataFrame X
1. Filter numeric columns only
2. Impute missing values using mean strategy
3. Remove zero-variance features
4. Store filtered column names for future transforms
```

### 2. Scaler Selection (_choose_scaler method)
```
Input: DataFrame X
For each feature column:
    If variance == 0:
        Mark as "Constant Feature" with MinMaxScaler
    Else:
        Sample up to 5000 data points
        Perform Shapiro-Wilk normality test
        If p_value > normality_ratio:
            Mark as "Normal" feature (StandardScaler)
        Else:
            Mark as "Non-normal" feature (MinMaxScaler)

Count normal vs non-normal features
Return StandardScaler if normal features > non-normal features
Otherwise return MinMaxScaler
```

### 3. PCA Component Selection (_evaluate_pca method)
```
Input: Scaled data matrix X_scaled
1. Determine maximum possible components:
   max_components = min(configured_max, n_samples, n_features)

2. Fit PCA with max_components
3. Calculate cumulative explained variance

For each n from 1 to max_components:
    If cumulative_variance[n-1] >= variance_threshold:
        Return {
            'best_scaler': current_scaler_name,
            'best_n_components': n,
            'best_explained_variance': cumulative_variance[n-1]
        }
Return None if no solution found
```

### 4. Data Transformation (transform method)
```
Input: DataFrame X
1. Filter numeric columns
2. Impute missing values using mean
3. Select only previously filtered columns
4. Apply stored scaler
5. Apply PCA with best_n_components
Return: Transformed data matrix
```

### 5. Validation (validate_with_classifier method)
```
Input: X, y, classifier (optional), cv (optional), test_size
Default classifier: LGBMClassifier

If cv is provided:
    Perform cross-validation on full dataset
    If PCA reduction successful:
        Perform cross-validation on reduced dataset
Else:
    Perform train-test split validation on full dataset
    If PCA reduction successful:
        Perform train-test split validation on reduced dataset

Calculate and display:
- Accuracy scores
- Processing times
- Efficiency gain percentage
```

## Key Features

### Error Handling
- Handles zero-variance features
- Manages missing values through mean imputation
- Validates presence of numeric columns
- Ensures fit before transform

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

## Complexity Analysis
- Time Complexity: O(n * d^2) for PCA computation
- Space Complexity: O(n * d) for data storage
Where n = samples, d = features

## Output Information
- Selected scaler type
- Optimal number of components
- Explained variance achieved
- Performance metrics comparison
- Processing time measurements
- Efficiency gains from dimensionality reduction


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

### Effect Size

Using Cohen's d and statistical tests, we observed significant effect sizes in processing time, favoring AdaptivePCA. In practical terms, this means that AdaptivePCA provides substantial improvements in performance while maintaining equivalent or higher levels of accuracy in explained variance coverage.

## Initialization Parameters
- `variance_threshold`: float, default=0.95  
  The cumulative variance explained threshold to determine the optimal number of components.
  
- `max_components`: int, default=50  
  The maximum number of components to consider.

- `variance_ratio`: Variance ratio threshold (default: 5.0)

- `normality_ratio`: P-value threshold for Shapiro-Wilk test (default: 0.05)

~~- `scaler_test`: bool, default=True  
  Added flexibility in scaling, which reduces runtime when scaling isn't required.
  Added on version 1.0.3~~

- `verbose`: int, default=0  
  Added parameter verbose to control the level of output:
   `verbose=1`: Provides detailed output, displaying all component-wise explained variance scores for each scaler.
   `verbose=0`: Suppresses intermediate output, showing only the final best configuration found after processing all scalers.
  Useful for debugging or fine-tuning PCA settings, with a default value of 0.
  Added on version 1.0.6

## Methods
- `fit(X)`: Fits the AdaptivePCA model to the data `X`.
- `transform(X)`: Transforms the data `X` using the fitted PCA model.
- `fit_transform(X)`: Fits and transforms the data in one step.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request to discuss your changes.

## Acknowledgments
This project makes use of the `scikit-learn`, `numpy`, and `pandas` libraries for data processing and machine learning.

## Version Update Log
- `1.0.3` - Added flexibility in scaling, fix error handling when max_components exceeding the available number of features or samples.
- `1.0.6` - Added Parameter verbose as an argument to __init__, with a default value of 0.

