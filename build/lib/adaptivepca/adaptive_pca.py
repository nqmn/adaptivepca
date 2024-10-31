"""
AdaptivePCA: An advanced PCA implementation with adaptive feature scaling and preprocessing.

This class implements Principal Component Analysis (PCA) with several enhancements:
- Automatic feature scaling based on data distribution
- Outlier handling
- Missing value imputation
- Class imbalance detection and handling via SMOTE
- Built-in validation with classifier support

- Version 1.1.3a:
  Add error handling for fit module
  Update export_model module
"""

import time
import json
import joblib
import numpy as np
import pandas as pd
from typing import Optional, Dict
from collections import Counter

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
                           roc_auc_score)
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier
from scipy.stats import shapiro, chi2_contingency
from imblearn.over_sampling import SMOTE

class AdaptivePCA:
    """
    AdaptivePCA: A comprehensive PCA implementation with adaptive preprocessing.
    
    This class provides an end-to-end solution for dimensionality reduction using PCA,
    with automatic feature scaling, preprocessing, and validation capabilities.
    
    Parameters:
    -----------
    variance_threshold : float, default=0.95
        The minimum cumulative explained variance ratio to determine optimal components
    max_components : int, default=50
        Maximum number of principal components to consider
    min_eigenvalue_threshold : float, default=1e-4
        Minimum eigenvalue threshold for component selection
    normality_ratio : float, default=0.05
        P-value threshold for Shapiro-Wilk normality test
    verbose : int, default=0
        Verbosity level for output logging
    """
    
    def __init__(self, variance_threshold: float = 0.95, 
                 max_components: int = 50,
                 min_eigenvalue_threshold: float = 1e-4, 
                 normality_ratio: float = 0.05, 
                 verbose: int = 0):
        # Initialize configuration parameters
        self.variance_threshold = variance_threshold
        self.max_components = max_components
        self.min_eigenvalue_threshold = min_eigenvalue_threshold
        self.normality_ratio = normality_ratio
        self.verbose = verbose
        
        # Initialize state variables
        self.scaler = None
        self.best_n_components = None
        self.best_explained_variance = None
        self.classifier = None
        
    def preprocess_data(self, X: pd.DataFrame, y: pd.Series, 
                       smote_test: bool = True) -> tuple[pd.DataFrame, pd.Series, bool]:
        """
        Comprehensive data preprocessing pipeline.
        
        Performs the following steps:
        1. Selects numeric columns
        2. Handles outliers using IQR method
        3. Replaces infinity values
        4. Imputes missing values
        5. Performs normality testing and selects appropriate scaler
        6. Applies SMOTE for class imbalance (if needed)
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
        y : pd.Series
            Target variable
        smote_test : bool, default=True
            Whether to test and apply SMOTE for class imbalance
            
        Returns:
        --------
        tuple[pd.DataFrame, pd.Series, bool]
            Preprocessed features, preprocessed target, and whether SMOTE was applied
        """
        if self.verbose:
            print("Starting data preprocessing steps...\n")
        
        # Step 1: Select numeric columns only
        X = X.select_dtypes(include=[np.number])
        if self.verbose:
            print("Step 1: Selected numeric columns.")
        
        # Step 2: Handle outliers using IQR method
        for col in X.columns:
            Q1, Q3 = X[col].quantile(0.25), X[col].quantile(0.75)
            IQR = Q3 - Q1
            X[col] = X[col].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)
        if self.verbose:
            print("Step 2: Handled outliers using IQR method.")

        # Step 3: Replace infinity values with finite extremes
        for col in X.columns:
            if np.isinf(X[col]).any():
                finite_vals = X[col][np.isfinite(X[col])]
                if not finite_vals.empty:
                    X[col] = X[col].replace([np.inf, -np.inf], 
                                          [finite_vals.max(), finite_vals.min()])
        if self.verbose:
            print("Step 3: Replaced infinity values with finite extremes.")
        
        # Step 4: Impute missing values using mean strategy
        self.imputer = SimpleImputer(strategy='mean')
        X = pd.DataFrame(self.imputer.fit_transform(X), columns=X.columns)
        if self.verbose:
            print("Step 4: Imputed missing values with column means.")
        
        # Step 5: Perform Shapiro-Wilk normality test for each feature
        feature_scaler_decisions = []
        if self.verbose:
            print("\nStep 5: Shapiro-Wilk Normality Test Results:")
            print("-" * 70)
            print(f"{'Feature':<20} {'P-Value':<10} {'Normality':<15} {'Scaler':<15}")
            print("-" * 70)
        
        for column in X.columns:
            if X[column].nunique() == 1:
                # Handle constant columns
                p_value = "N/A"
                normality = "Constant"
                scaler_choice = "MinMaxScaler()"
            else:
                # Perform normality test
                _, p_value_val = shapiro(X[column].sample(min(5000, len(X[column]))))
                p_value = f"{p_value_val:.4f}"
                normality = 'Normal' if p_value_val > self.normality_ratio else 'Non-normal'
                scaler_choice = 'StandardScaler()' if normality == 'Normal' else 'MinMaxScaler()'
            
            feature_scaler_decisions.append({
                'Feature': column,
                'P-Value': p_value,
                'Normality': normality,
                'Scaler': scaler_choice
            })
            
            if self.verbose:
                print(f"{column:<20} {p_value:<10} {normality:<15} {scaler_choice:<15}")
        
        # Step 6: Select and apply final scaler based on majority rule
        num_normal_features = sum(1 for decision in feature_scaler_decisions 
                                if decision['Scaler'] == 'StandardScaler()')
        num_non_normal_features = len(feature_scaler_decisions) - num_normal_features
        
        # Choose scaler based on majority of feature distributions
        self.scaler = StandardScaler() if num_normal_features > num_non_normal_features else MinMaxScaler()
        X = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)
        
        if self.verbose:
            chosen_scaler = self.scaler.__class__.__name__
            print(f"\nStep 6: Final Scaler Decision")
            print(f"P-value threshold: {self.normality_ratio}")
            print(f"Normal features count: {num_normal_features}")
            print(f"Non-normal features count: {num_non_normal_features}")
            print(f"Chosen Scaler: {chosen_scaler}")

        # Fit initial PCA to determine optimal components
        pca = PCA(n_components=min(self.max_components, X.shape[1]))
        pca.fit(X)
        eigenvalues = pca.explained_variance_
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        
        # Determine optimal number of components based on variance threshold
        self.best_n_components = next(
            (i + 1 for i, explained_variance in enumerate(cumulative_variance) 
             if explained_variance >= self.variance_threshold), 
            min(self.max_components, X.shape[1])
        )
        
        # Log PCA component analysis if verbose
        if self.verbose:
            print("\nComponent Explained Variance and Eigenvalue Progression:")
            print("-" * 70)
            print(f"{'Scaler':<20}{'Component':<15}{'Eigenvalue':<15}{'Cumulative Variance':<15}")
            print("-" * 70)
            
            for i, (eigenvalue, explained_variance) in enumerate(zip(eigenvalues, cumulative_variance), 1):
                print(f"{self.scaler.__class__.__name__:<20}{i:<15}{eigenvalue:<15.6f}"
                      f"{explained_variance:<15.6f}")
                if i >= self.best_n_components:
                    break

        # Step 7: Handle class imbalance with SMOTE if needed
        smote_applied = False
        
        # Perform chi-squared test for class imbalance
        counts = Counter(y)
        obs = list(counts.values())
        total_count = sum(obs)
        expected = [total_count / len(obs)] * len(obs)
        chi2, p_value, _, _ = chi2_contingency([obs, expected])
        
        if self.verbose:
            print(f"\nStep 7: Chi-squared test p-value for class imbalance: {p_value:.6f}")
        
        # Apply SMOTE if imbalance detected and enabled
        if p_value < 0.05 and smote_test:
            if self.verbose:
                print("Class imbalance detected; applying SMOTE.")
                print(f"Data length before SMOTE: {len(X)}")
            
            smote = SMOTE(random_state=42)
            X, y = smote.fit_resample(X, y)
            smote_applied = True
            
            if self.verbose:
                print(f"Data length after SMOTE: {len(X)}")
        elif p_value > 0.05 and smote_test:
            if self.verbose:
                print("Class distribution is balanced; skipping SMOTE.")
        else:
            if self.verbose:
                print("SMOTE application skipped by user.")
        
        return X, y, smote_applied

    def _choose_scaler(self, X: pd.DataFrame):
        """
        Internal method to choose appropriate scaler based on feature normality.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features to analyze for scaling decision
            
        Returns:
        --------
        sklearn scaler
            Either StandardScaler or MinMaxScaler based on feature distributions
        """
        feature_scaler_decisions = []
        
        for column in X.columns:
            if X[column].nunique() == 1:
                # Skip normality test for constant columns
                feature_scaler_decisions.append('MinMaxScaler()')
            else:
                # Perform Shapiro-Wilk test on sample
                _, p_value = shapiro(X[column].sample(min(5000, len(X[column]))))
                scaler_choice = ('StandardScaler()' 
                               if p_value > self.normality_ratio 
                               else 'MinMaxScaler()')
                feature_scaler_decisions.append(scaler_choice)
        
        # Choose scaler based on majority vote
        num_standard = feature_scaler_decisions.count('StandardScaler()')
        return (StandardScaler() 
                if num_standard > len(feature_scaler_decisions) / 2 
                else MinMaxScaler())

    def fit(self, X: pd.DataFrame, y: pd.Series, smote_applied: bool):
        """
        Fit the AdaptivePCA model to the data.
        
        This method:
        1. Applies the chosen scaler
        2. Fits PCA with the optimal number of components
        3. Records performance metrics
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
        y : pd.Series
            Target variable
        smote_applied : bool
            Whether SMOTE was applied in preprocessing
            
        Returns:
        --------
        dict
            Configuration and performance metrics
        """
        
        # Ensure that self.scaler is set
        if not self.scaler:
            raise RuntimeError("Scaler is not set. Run preprocess_data before fit.")
        
        # Track fitting time
        start_time = time.time()
        
        # Apply scaling and PCA
        X_scaled = self.scaler.transform(X)
        pca = PCA(n_components=self.best_n_components)
        X_reduced = pca.fit_transform(X_scaled)
        self.best_explained_variance = np.cumsum(pca.explained_variance_ratio_)[-1]
        
        # Calculate fitting time
        fit_time = time.time() - start_time
        
        # Prepare results summary
        result = {
            "scaler": ("MinMaxScaler()" 
                      if isinstance(self.scaler, MinMaxScaler) 
                      else "StandardScaler()"),
            "smote": "yes" if smote_applied else "no",
            "pca_optimal_components": int(self.best_n_components),
            "score": round(float(self.best_explained_variance), 6),
            "fit_time_seconds": round(float(fit_time), 4)
        }
        
        # Store JSON output
        self.fit_result_json = json.dumps(result, indent=4)
        
        # Display results based on verbosity
        if self.verbose:
            print("\nBest configuration found:")
            print(f"{'Scaler':<20}{'Components':<15}{'Score':<15}{'Time (s)':<15}")
            print("=" * 70)
            print(f"{self.scaler.__class__.__name__:<20}"
                  f"{self.best_n_components:<15}"
                  f"{self.best_explained_variance:<15.4f}"
                  f"{fit_time:.4f}")
            print("=" * 70)
        else:
            print(self.fit_result_json)
        
        return result

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform data using the fitted scaler and PCA model.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features to transform
            
        Returns:
        --------
        np.ndarray
            Transformed data with reduced dimensions
            
        Raises:
        -------
        RuntimeError
            If model hasn't been fitted
        """
        # Verify model has been fitted
        if not self.best_n_components:
            raise RuntimeError("Model must be fitted before calling transform.")
        
        # Apply scaling and PCA transformation
        X_scaled = self.scaler.transform(X)
        pca = PCA(n_components=self.best_n_components)
        X_reduced = pca.fit_transform(X_scaled)
        
        return X_reduced

    def validate_with_classifier(self, X: pd.DataFrame, y: pd.Series, 
                            classifier=None, cv: Optional[int] = 5, 
                            test_size: Optional[float] = 0.2):
        """
        Validate the model using a classifier and cross-validation or train-test split.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
        y : pd.Series
            Target variable
        classifier : estimator object, optional
            Classifier to use for validation (defaults to LGBMClassifier)
        cv : int, optional
            Number of cross-validation folds (if None, uses train-test split)
        test_size : float, optional
            Test size for train-test split (used only if cv is None)
            
        Returns:
        --------
        classifier
            Fitted classifier object
        """
        # Initialize classifier if not provided
        self.classifier = classifier or LGBMClassifier(verbosity=-1)
        
        # Validate on full dataset
        start_full = time.time()
        if cv:
            full_data_scores = cross_val_score(self.classifier, X, y, cv=cv)
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            self.classifier.fit(X_train, y_train)
            full_data_scores = [self.classifier.score(X_test, y_test)]
        end_full = time.time()
        
        mean_accuracy_full = np.mean(full_data_scores)
        time_full = end_full - start_full
        
        # Validate on PCA-reduced dataset if applicable
        if self.best_n_components:
            X_reduced = self.transform(X)
            start_pca = time.time()
            
            if cv:
                reduced_data_scores = cross_val_score(
                    self.classifier, X_reduced, y, cv=cv
                )
                # Fit final model on full reduced dataset
                self.classifier.fit(X_reduced, y)
            else:
                X_train_reduced, X_test_reduced, y_train, y_test = train_test_split(
                    X_reduced, y, test_size=test_size, random_state=42
                )
                self.classifier.fit(X_train_reduced, y_train)
                reduced_data_scores = [self.classifier.score(X_test_reduced, y_test)]
                
            end_pca = time.time()

            # Store the fitted classifier for future use in predict_with_classifier
            #self.classifier = classifier
            
            mean_accuracy_pca = np.mean(reduced_data_scores)
            time_pca = end_pca - start_pca
            efficiency_gain = ((time_full - time_pca) / time_full) * 100 if time_full > 0 else 0
            
            # Prepare validation results
            result = {
                "test": "cv" if cv else "split_test",
                "cv" if cv else "test_size": cv if cv else test_size,
                "full_dataset_score": round(mean_accuracy_full, 6),
                "full_dataset_time_seconds": round(time_full, 4),
                "reduced_dataset_score": round(mean_accuracy_pca, 6),
                "reduced_dataset_time_seconds": round(time_pca, 4),
                "efficiency_gain_percent": round(efficiency_gain, 2)
            }
            
            # Store and display results
            self.validation_result_json = json.dumps(result, indent=4)
            
            if self.verbose:
                print("\nDataset Performance:")
                print("-" * 75)
                print(f"{'Dataset':<20}{'(CV)' if cv else '(Split-test)':<15}"
                      f"{'Score (Acc)':<15}{'Time (s)':<15}{'Gain (%)':<15}")
                print("-" * 75)
                print(f"{'Full Dataset':<20}{cv if cv else test_size:<15}"
                      f"{mean_accuracy_full:<15.6f}{time_full:<15.4f}")
                print(f"{'Reduced Dataset':<20}{cv if cv else test_size:<15}"
                      f"{mean_accuracy_pca:<15.6f}{time_pca:<15.4f}{efficiency_gain:.2f}")
                print("-" * 75)
            else:
                print(self.validation_result_json)
                
        return classifier

    def predict_with_classifier(self, X: pd.DataFrame, y: pd.Series):
        """
        Make predictions using the fitted classifier and evaluate performance.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
        y : pd.Series
            True target values for evaluation
            
        Returns:
        --------
        dict
            Dictionary containing performance metrics
            
        Raises:
        -------
        RuntimeError
            If classifier hasn't been fitted
        """
        # Verify classifier has been fitted
        if not hasattr(self.classifier, 'classes_'):
            raise RuntimeError(
                "Classifier must be fitted first. Run validate_with_classifier."
            )
        
        # Transform data if PCA was applied
        X_transformed = self.transform(X) if self.best_n_components else X
        
        # Make predictions and measure inference time
        start_time = time.time()
        y_pred = self.classifier.predict(X_transformed)
        inference_time = time.time() - start_time
        
        # Calculate performance metrics
        confusion = confusion_matrix(y, y_pred)
        class_report = classification_report(y, y_pred, digits=6, output_dict=True)
        roc_auc = roc_auc_score(y, y_pred)
        
        # Round classification report metrics
        class_report = {
            key: ({metric: round(value, 6) for metric, value in scores.items()}
                  if isinstance(scores, dict) else round(scores, 6))
            for key, scores in class_report.items()
        }
        
        # Prepare results summary
        result = {
            "confusion_matrix": confusion.tolist(),
            "classification_report": class_report,
            "roc_auc_score": round(roc_auc, 6),
            "inference_time_seconds": round(inference_time, 4)
        }
        
        # Store JSON output
        self.predict_result_json = json.dumps(result, indent=4)
        
        # Display results based on verbosity
        if self.verbose:
            print("\nPrediction Results:")
            print("Confusion Matrix:\n", confusion)
            print("\nClassification Report:\n", 
                  classification_report(y, y_pred, digits=6))
            print(f"ROC-AUC Score: {roc_auc:.6f}")
            print(f"Inference Time: {inference_time:.4f} seconds")
        else:
            print(self.predict_result_json)
        
        #return result

    def export_model(self, model_name: str) -> None:
        """
        Export the entire AdaptivePCA model, including PCA configuration,
        fitted scaler, fitted PCA instance, and fitted classifier, to a file using joblib.
        
        Parameters:
        model_name (str): The filename for the exported model.
        classifier: A fitted LightGBM classifier to include in the model export.
        """
        # Verify that the classifier is fitted
        if not hasattr(self.classifier, "classes_"):
            raise ValueError("Classifier must be fitted before exporting. Run validate_with_classifier.")

        # Save the full state of the model, including fitted scaler, PCA instance, and classifier
        model_data = {
            'variance_threshold': self.variance_threshold,
            'max_components': self.max_components,
            'min_eigenvalue_threshold': self.min_eigenvalue_threshold,
            'normality_ratio': self.normality_ratio,
            'verbose': self.verbose,
            'best_scaler': self.scaler,
            'best_n_components': self.best_n_components,
            'best_explained_variance': self.best_explained_variance,
            'scaler': self.scaler,  # Fitted scaler
            'pca': getattr(self, 'pca', None),  # Fitted PCA instance
            'classifier': self.classifier,  # Fitted classifier
        }
        
        # Save the model data with joblib
        joblib.dump(model_data, model_name)
        print(f"Model exported successfully to {model_name}")
        