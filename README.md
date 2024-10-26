
# AdaptivePCA

AdaptivePCA is a Python package designed for high-performance Principal Component Analysis (PCA) with an adaptive component selection approach. It allows the user to perform dimensionality reduction on large datasets efficiently, choosing the optimal number of PCA components based on a specified explained variance threshold. It supports both StandardScaler and MinMaxScaler for data preprocessing and can operate in both parallel and non-parallel modes.

## Features
- **Automatic Component Selection**: Automatically chooses the number of components needed to reach a specified variance threshold.
- **Scaler Options**: Supports StandardScaler and MinMaxScaler for data scaling.
- **Parallel Processing**: Uses parallel processing to speed up computations, particularly beneficial for large datasets.
- **Easy Integration**: Designed to integrate seamlessly with other data science workflows.

## Installation

Clone this repository and install the package using `pip`:
```bash
git clone https://github.com/yourusername/adaptivepca.git
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
y = data['Label']  # Target variable

# Initialize and fit AdaptivePCA
adaptive_pca = AdaptivePCA(variance_threshold=0.95, max_components=10)
X_reduced = adaptive_pca.fit_transform(X)

# Results
print("Optimal Components:", adaptive_pca.best_n_components)
print("Explained Variance:", adaptive_pca.best_explained_variance)
```

## Parameters
- `variance_threshold`: float, default=0.95  
  The cumulative variance explained threshold to determine the optimal number of components.
  
- `max_components`: int, default=10  
  The maximum number of components to consider.

## Methods
- `fit(X)`: Fits the AdaptivePCA model to the data `X`.
- `transform(X)`: Transforms the data `X` using the fitted PCA model.
- `fit_transform(X)`: Fits and transforms the data in one step.

## Example

Below is an example usage of AdaptivePCA in parallel mode:

```python
adaptive_pca = AdaptivePCA(variance_threshold=0.95, max_components=10)
X_reduced = adaptive_pca.fit_transform(X)

print(f"Optimal scaler: {adaptive_pca.best_scaler}")
print(f"Number of components: {adaptive_pca.best_n_components}")
print(f"Explained variance: {adaptive_pca.best_explained_variance}")
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request to discuss your changes.

## Acknowledgments
This project makes use of the `scikit-learn`, `numpy`, and `pandas` libraries for data processing and machine learning.
