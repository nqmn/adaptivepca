AdaptivePCA
AdaptivePCA is a flexible, scalable Python package that enables dimensionality reduction with PCA, automatically selecting the best scaler and the optimal number of components to meet a specified variance threshold. Built for efficiency, AdaptivePCA includes parallel processing capabilities to speed up large-scale data transformations, making it ideal for data scientists and machine learning practitioners working with high-dimensional datasets.

Features
Automatic Component Selection: Automatically selects the optimal number of principal components based on a specified variance threshold.
Scaler Selection: Compares multiple scalers (StandardScaler and MinMaxScaler) to find the best fit for the data.
Parallel Processing: Option to use concurrent scaling for faster computations.
Easy Integration: Built on top of widely-used libraries like scikit-learn and numpy.
Installation
You can install AdaptivePCA via pip:

bash
Copy code
pip install adaptivepca
Usage
Import and Initialize
python
Copy code
from adaptivepca import AdaptivePCA
import pandas as pd

# Load your dataset
X = pd.read_csv("your_data.csv")  # Ensure your dataset is loaded as a Pandas DataFrame
Basic Usage
Initialize AdaptivePCA and fit it to your data:

python
Copy code
# Initialize AdaptivePCA with desired variance threshold and maximum components
adaptive_pca = AdaptivePCA(variance_threshold=0.95, max_components=10)

# Fit and transform data
X_transformed = adaptive_pca.fit_transform(X)
Parallel Processing
For larger datasets, enable parallel processing to speed up computations:

python
Copy code
# Fit AdaptivePCA with parallel processing
adaptive_pca.fit(X, parallel=True)
Accessing Best Parameters
After fitting, you can retrieve the best scaler, number of components, and explained variance score:

python
Copy code
print(f"Best Scaler: {adaptive_pca.best_scaler}")
print(f"Optimal Components: {adaptive_pca.best_n_components}")
print(f"Explained Variance Score: {adaptive_pca.best_explained_variance}")
Parameters
variance_threshold (float): Desired variance threshold for component selection. Default is 0.95.
max_components (int): Maximum number of PCA components to consider. Default is 10.
Methods
fit(X, parallel=False): Fits AdaptivePCA to the dataset X. Use parallel=True to enable parallel processing.
transform(X): Transforms the dataset X using the previously fitted configuration.
fit_transform(X): Combines fit and transform steps in one call.
Example
python
Copy code
from adaptivepca import AdaptivePCA
import pandas as pd

# Example dataset
X = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5],
    'feature2': [10, 9, 8, 7, 6],
    'feature3': [2, 4, 6, 8, 10]
})

adaptive_pca = AdaptivePCA(variance_threshold=0.95, max_components=2)
X_transformed = adaptive_pca.fit_transform(X)

# Retrieve best configuration details
print(f"Best Scaler: {adaptive_pca.best_scaler}")
print(f"Optimal Components: {adaptive_pca.best_n_components}")
print(f"Explained Variance Score: {adaptive_pca.best_explained_variance}")
Dependencies
scikit-learn>=0.24
numpy>=1.19
pandas>=1.1
License
This project is licensed under the MIT License. See the LICENSE file for details.