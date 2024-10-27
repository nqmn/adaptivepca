from setuptools import setup, find_packages

setup(
    name="adaptivepca",
    version="1.0.3",
    description="Adaptive PCA with parallel scaling and dimensionality reduction",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Mohd Adil",
    author_email="mohdadil@live.com",
    url="https://github.com/nqmn/adaptivepca",
    packages=find_packages(),
    install_requires=[
        "scikit-learn>=0.24",
        "numpy>=1.19",
        "pandas>=1.1"
    ],
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
