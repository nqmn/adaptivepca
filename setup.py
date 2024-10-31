from setuptools import setup, find_packages

setup(
    name="adaptivepca",
    version="1.1.3",
    description="An advanced PCA implementation with adaptive feature scaling and preprocessing",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Mohd Adil",
    author_email="mohdadil@live.com",
    url="https://github.com/nqmn/adaptivepca",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "lightgbm>=3.3.0",
        "imbalanced-learn>=0.8.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
    # Keywords for PyPI
    keywords=[
        "machine learning",
        "dimensionality reduction",
        "pca",
        "feature selection",
        "data preprocessing",
        "adaptive scaling",
        "classification",
        "data analysis",
        "statistics",
    ],

)
