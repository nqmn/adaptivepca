
# AdaptivePCA

AdaptivePCA is a Python library introducing a novel approach to optimize Principal Component Analysis (PCA) for large datasets through an adaptive mechanism. By intelligently selecting the most suitable scaling method—choosing between StandardScaler and MinMaxScaler—and combining it with PCA, AdaptivePCA innovatively preserves the most significant information while reducing dimensionality. With a unique auto-stop feature that halts component selection upon reaching the specified variance threshold, AdaptivePCA ensures computational efficiency. This adaptive selection strategy enables efficient identification of the optimal principal components, positioning AdaptivePCA as a unique tool to enhance machine learning model performance and streamline data visualization tasks.

## Features
- **Adaptive Scaling Selection**: Dynamically selects between StandardScaler and MinMaxScaler to identify the most effective scaling method, optimizing information retention during dimensionality reduction.
- **Automatic Component Optimization**: Automatically adjusts the number of principal components to achieve a specified variance threshold, preserving maximum data variance with minimal components.
- **Efficient Parallel Processing**: Leverages parallel computation to accelerate scaling and component evaluation, enhancing performance on large datasets.
- **Auto-Stop for Efficiency**: Stops further component evaluation once the specified variance threshold is reached, making the process computationally efficient.
- **Seamless Integration**: Easily integrates into data science workflows, enhancing compatibility with machine learning pipelines and data visualization tasks.

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
adaptive_pca = AdaptivePCA(variance_threshold=0.95, max_components=50, scaler_test=True)
X_reduced = adaptive_pca.fit_transform(X)
```

## Performance Comparison: AdaptivePCA vs. Traditional PCA

### Speed

AdaptivePCA leverages parallel processing to evaluate scaling and PCA component selection concurrently. In our tests, AdaptivePCA achieved up to a 95% reduction in processing time compared to the traditional PCA method. This is especially useful when working with high-dimensional data, where traditional methods may take significantly longer due to sequential grid search.

### Explained Variance

Both AdaptivePCA and traditional PCA achieve similar levels of explained variance, with AdaptivePCA dynamically selecting the number of components based on a defined variance threshold. Traditional PCA, on the other hand, requires manual parameter tuning, which can be time-consuming.

### Effect Size

Using Cohen's d and statistical tests, we observed significant effect sizes in processing time, favoring AdaptivePCA. In practical terms, this means that AdaptivePCA provides substantial improvements in performance while maintaining equivalent or higher levels of accuracy in explained variance coverage.

## Parameters
- `variance_threshold`: float, default=0.95  
  The cumulative variance explained threshold to determine the optimal number of components.
  
- `max_components`: int, default=10  
  The maximum number of components to consider. Set to 50 for comprehensive evaluation.

- `scaler_test`: bool, default=True  
  Added flexibility in scaling, which reduces runtime when scaling isn't required.
  Added on version 1.0.3

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
- `1.0.3` - Add flexibility in scaling, fix error handling when max_components exceeding the available number of features or samples.
