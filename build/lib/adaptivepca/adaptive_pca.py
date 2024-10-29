# Version 1.1.0 Pypi
# Cleanup the code

import time, joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, matthews_corrcoef, auc, roc_curve
from lightgbm import LGBMClassifier
from scipy.stats import shapiro
import numpy as np
import pandas as pd
from typing import Optional, Dict
from sklearn.impute import SimpleImputer

class AdaptivePCA:
    def __init__(self, variance_threshold: float = 0.95, max_components: int = 50, variance_ratio: float = 5.0, 
                 normality_ratio: float = 0.05, verbose: int = 0):
        self.variance_threshold = variance_threshold
        self.max_components = max_components
        self.variance_ratio = variance_ratio
        self.normality_ratio = normality_ratio
        self.verbose = verbose
        self.best_scaler = None
        self.best_n_components = None
        self.best_explained_variance = None
        self.scaler = None
        self.filtered_columns = None  # Store filtered column names

    def _preprocess_data(self, X: pd.DataFrame, fit_stage: bool = False) -> pd.DataFrame:
        """
        Preprocess data by selecting numeric columns, imputing missing values,
        and filtering based on variance. The 'fit_stage' flag is used to differentiate
        between fit and transform stages.
    
        Parameters:
        X (pd.DataFrame): Input data to preprocess.
        fit_stage (bool): Whether the function is called during fit or transform.
    
        Returns:
        pd.DataFrame: Processed data ready for scaling and PCA.
        """
        # Keep only numeric columns
        X = X.select_dtypes(include=[np.number])

        # Replace infinity values with NaN to impute later
        #X = X.replace([np.inf, -np.inf], np.nan)

        # Replace infinity values with column-specific max/min finite values
        for col in X.columns:
            finite_vals = X[col][np.isfinite(X[col])]  # Only finite values
            if not finite_vals.empty:
                max_val, min_val = finite_vals.max(), finite_vals.min()
                X[col] = X[col].replace(np.inf, max_val).replace(-np.inf, min_val)  # Replace infinities

        # Drop rows with infinity values
        # X = X[~X.isin([np.inf, -np.inf]).any(axis=1)]
    
        # Impute missing values
        imputer = SimpleImputer(strategy='mean')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
        # During fit, store columns with non-zero variance for later use in transform
        if fit_stage:
            X = X.loc[:, (X.var() > 0)]
            self.filtered_columns = X.columns
        else:
            # During transform, use stored columns from the fit stage
            X = X[self.filtered_columns]
    
        return X

    def _choose_scaler(self, X: pd.DataFrame):
        variances = X.var()
        non_zero_variances = variances[variances > 0]
        
        if non_zero_variances.empty:
            print("All features have zero variance. Defaulting to StandardScaler.")
            return StandardScaler()
        
        # Adding a column-wise normality check with a check for zero variance
        feature_scaler_decisions = []
        for column in X.columns:
            if X[column].var() == 0:
                # Skip Shapiro-Wilk test for constant features to avoid warnings
                feature_scaler_decisions.append({
                    'Feature': column,
                    'P-Value': None,
                    'Normality': 'Constant Feature',
                    'Scaler': 'MinMaxScaler'  # Default choice for non-varying data
                })
            else:
                # Perform Shapiro-Wilk test for normality on non-constant features
                stat, p_value = shapiro(X[column].sample(min(5000, len(X[column]))))  # Sample up to 5000 for efficiency
                normality = 'Normal' if p_value > self.normality_ratio else 'Non-normal'
                scaler_choice = 'StandardScaler' if normality == 'Normal' else 'MinMaxScaler'
                
                # Display p-value as "<0.0001" for very low values
                p_value_display = f"{p_value:.4f}" if p_value >= 1e-4 else "<0.0001"
                
                feature_scaler_decisions.append({
                    'Feature': column,
                    'P-Value': p_value_display,
                    'Normality': normality,
                    'Scaler': scaler_choice
                })
    
        # Display normality results in a readable format
        if self.verbose == 1:
            print("\nFeature Normality and Scaler Decision based on Shapiro-Wilk test:\n")
            print(f"{'Feature':<20}{'P-Value':<12}{'Normality':<15}{'Scaler':<15}")
            print("-" * 60)
            for decision in feature_scaler_decisions:
                print(f"{decision['Feature']:<20}{decision['P-Value']:<12}{decision['Normality']:<15}{decision['Scaler']:<15}")
        
        # Choose final scaler based on overall results
        num_normal_features = sum(1 for d in feature_scaler_decisions if d['Scaler'] == 'StandardScaler')
        num_non_normal_features = len(feature_scaler_decisions) - num_normal_features
        return StandardScaler() if num_normal_features > num_non_normal_features else MinMaxScaler()

    def _apply_scaler(self, X: pd.DataFrame) -> np.ndarray:
        return self.scaler.fit_transform(X) if self.scaler else X

    def _evaluate_pca(self, X_scaled: np.ndarray, scaler_name: str) -> Optional[Dict[str, float]]:
        max_components = min(self.max_components, X_scaled.shape[0], X_scaled.shape[1])
        pca = PCA(n_components=max_components)
        pca.fit(X_scaled)
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

        # Display PCA component scores after final scaler decision
        if self.verbose == 1:
            print("-" * 50)
            print(f"{'Scaler':<20}{'Components':<12}{'Score':<12}")
            print("-" * 50)
        
        for n_components in range(1, max_components + 1):
            explained_variance_score = cumulative_variance[n_components - 1]
            if self.verbose == 1:
                self._log_test_result(scaler_name, n_components, explained_variance_score)

            if explained_variance_score >= self.variance_threshold:
                return {
                    'best_scaler': scaler_name,
                    'best_n_components': n_components,
                    'best_explained_variance': explained_variance_score
                }
        return None

    def _log_test_result(self, scaler_name: str, n_components: int, score: float):
        print(f"{scaler_name:<20}{n_components:<12}{score:<12.6f}")

    def fit(self, X: pd.DataFrame):
        # Preprocess data for fitting stage
        X = self._preprocess_data(X, fit_stage=True)
        
        # Track the start time for the fitting process
        start_time = time.time()
    
        # Determine the best scaler using normality tests
        self.scaler = self._choose_scaler(X)
        scaler_name = 'StandardScaler' if isinstance(self.scaler, StandardScaler) else 'MinMaxScaler'
    
        # Apply scaler and evaluate PCA
        X_scaled = self._apply_scaler(X)
        best_config = self._evaluate_pca(X_scaled, scaler_name)
    
        # Set the best configuration
        if best_config:
            self.best_scaler = best_config['best_scaler']
            self.best_n_components = best_config['best_n_components']
            self.best_explained_variance = best_config['best_explained_variance']
        else:
            print("Warning: No configuration met the variance threshold. Setting components to 0.")
            self.best_scaler = 'None'
            self.best_n_components = 0
            self.best_explained_variance = 0.0
    
        # Display final results with elapsed time
        self._display_final_results(time.time() - start_time)

    def _display_final_results(self, elapsed_time: float):
        print("-" * 70)
        print("Best configuration found:" if self.best_n_components > 0 else "No suitable configuration found.")
        print("=" * 70)
        print(f"{'Best Scaler':<20}{'Optimal Components':<20}{'Best Score':<15}{'Time (s)':<15}")
        print("-" * 70)
        print(f"{self.best_scaler:<20}{self.best_n_components:<20}{self.best_explained_variance:<15.6f}{elapsed_time:.4f}")
        print("=" * 70)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self.best_n_components:
            raise RuntimeError("You must fit the AdaptivePCA model before calling transform.")

        # Preprocess data for transform stage
        X = self._preprocess_data(X, fit_stage=False)

        # Apply scaler and transform with fitted PCA
        X_scaled = self.scaler.transform(X)  # Use the stored scaler
        pca = PCA(n_components=self.best_n_components)
        return pca.fit_transform(X_scaled)

    def fit_transform(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        self.fit(X)
        return self.transform(X) if self.best_n_components else None

    def export_model(self, model_name: str, classifier) -> None:
        """
        Export the entire AdaptivePCA model, including PCA configuration,
        fitted scaler, fitted PCA instance, and fitted classifier, to a file using joblib.
        
        Parameters:
        model_name (str): The filename for the exported model.
        classifier: A fitted LightGBM classifier to include in the model export.
        """
        # Verify that the classifier is fitted
        if not hasattr(classifier, "classes_"):
            raise ValueError("The classifier must be fitted before exporting.")

        # Save the full state of the model, including fitted scaler, PCA instance, and classifier
        model_data = {
            'variance_threshold': self.variance_threshold,
            'max_components': self.max_components,
            'variance_ratio': self.variance_ratio,
            'normality_ratio': self.normality_ratio,
            'verbose': self.verbose,
            'best_scaler': self.best_scaler,
            'best_n_components': self.best_n_components,
            'best_explained_variance': self.best_explained_variance,
            'scaler': self.scaler,  # Fitted scaler
            'pca': getattr(self, 'pca', None),  # Fitted PCA instance
            'classifier': classifier,  # Fitted classifier
            'filtered_columns': self.filtered_columns
        }
        
        # Save the model data with joblib
        joblib.dump(model_data, model_name)
        print(f"Model exported successfully to {model_name}")

    def view_config(self):
        """
        Display the current configuration and attributes of the AdaptivePCA instance.
        """
        print("AdaptivePCA Configuration:")
        print("-" * 40)
        print("Variance Threshold:", self.variance_threshold)
        print("Max Components:", self.max_components)
        print("Variance Ratio:", self.variance_ratio)
        print("Normality Ratio:", self.normality_ratio)
        print("Verbose:", self.verbose)
        print("Best Scaler:", self.best_scaler)
        print("Best Number of Components:", self.best_n_components)
        print("Best Explained Variance:", self.best_explained_variance)
        print("Scaler:", self.scaler)
        print("Filtered Columns:", self.filtered_columns)
        print("-" * 40)

    def validate_with_classifier(self, X: pd.DataFrame, y: pd.Series, classifier=None, cv: Optional[int] = 5, test_size: float = 0.2):
        if classifier is None:
            classifier = LGBMClassifier(verbosity=-1)

        # Preprocess data for transform stage
        X = self._preprocess_data(X, fit_stage=False)
        
        # Ensure X contains only numeric columns before fitting the classifier
        #X = X.select_dtypes(include=[np.number])

        # Impute missing values
        #imputer = SimpleImputer(strategy='mean')
        #X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        
        # Full dataset evaluation with either cross-validation or split-test
        if cv is None:
            # Split-test method when cv is None
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            start_full = time.time()
            
            # Train on the training set
            classifier.fit(X_train, y_train)
            full_data_score = classifier.score(X_test, y_test)  # Evaluate on the test set
            end_full = time.time()
            
            # Store result as a list for compatibility in later calculations
            full_data_scores = [full_data_score]
        else:
            # Cross-validation method
            start_full = time.time()
            full_data_scores = cross_val_score(classifier, X, y, cv=cv)
            end_full = time.time()
        
        # Calculate timing and display performance
        mean_accuracy_full = np.mean(full_data_scores)
        time_full = end_full - start_full
        print("\nDataset Performance:\n")
        print("Model: ", classifier)
        print("-" * 75)
        print(f"{'Dataset':<20}{'(CV)' if cv else '(Split-test)':<15}{'Score (Acc)':<15}{'Time (s)':<15}{'Gain (%)':<15}")
        print("-" * 75)
        print(f"{'Full Dataset':<20}{cv if cv is not None else test_size:<15}{mean_accuracy_full:<15.6f}{time_full:<15.4f}")
        
        # Reduced dataset cross-validation or split-test with timing
        if self.best_n_components:
            X_reduced = self.transform(X)

            # Split-test for the reduced dataset
            X_train_reduced, X_test_reduced, y_train, y_test = train_test_split(X_reduced, y, test_size=test_size, random_state=42)
            
            if cv is None:
                start_pca = time.time()
                
                # Train on the reduced training set
                classifier.fit(X_train_reduced, y_train)
                reduced_data_score = classifier.score(X_test_reduced, y_test)  # Evaluate on the reduced test set
                end_pca = time.time()
                
                # Store result as a list for compatibility
                reduced_data_scores = [reduced_data_score]
            else:
                # Cross-validation on the reduced dataset
                start_pca = time.time()
                reduced_data_scores = cross_val_score(classifier, X_reduced, y, cv=cv)
                end_pca = time.time()

                # Fit the classifier on the reduced dataset after cross-validation
                classifier.fit(X_train_reduced, y_train)
                
            # Store the fitted classifier for future use in predict_with_classifier
            self.classifier = classifier
            
            # Calculate timing and display performance
            mean_accuracy_pca = np.mean(reduced_data_scores)
            time_pca = end_pca - start_pca
    
            # Calculate and display efficiency gain
            efficiency_gain = ((time_full - time_pca) / time_full) * 100 if time_full > 0 else 0
            
            print(f"{'Reduced Dataset':<20}{cv if cv is not None else test_size:<15}{mean_accuracy_pca:<15.6f}{time_pca:<15.4f}{efficiency_gain:.2f}")
            print("-" * 75)
        else:
            print("\nNo PCA reduction was applied, skipping PCA-reduced dataset validation.")

        return classifier

    def predict_with_classifier(self, X: pd.DataFrame, y: pd.Series):
        """
        Use the fitted classifier to make predictions on the reduced dataset and display
        confusion matrix, classification report, and inference time.

        Parameters:
        X (pd.DataFrame): The features for prediction.
        y_true (pd.Series): The true labels for the dataset.

        Raises:
        RuntimeError: If the classifier is not fitted, prompting to run validate_with_classifier first.
        """
        # Check if the classifier is fitted
        if not hasattr(self, 'classifier') or not hasattr(self.classifier, 'classes_'):
            raise RuntimeError("The classifier must be fitted before using predict_with_classifier. "
                               "Please run validate_with_classifier first.")

        # Preprocess data for transform stage
        X = self._preprocess_data(X, fit_stage=False)
        
        # Preprocess and transform the dataset
        #X = X.select_dtypes(include=[np.number])
        #imputer = SimpleImputer(strategy='mean')
        #X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

        # Apply PCA reduction
        X_reduced = self.transform(X)

        # Measure inference time
        start_time = time.time()
        y_pred = self.classifier.predict(X_reduced)
        end_time = time.time()
        inference_time = end_time - start_time

        # Confusion matrix and derived metrics
        conf_matrix = confusion_matrix(y, y_pred)
        if conf_matrix.shape == (2, 2):  # Binary classification
            tn, fp, fn, tp = conf_matrix.ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
            far = fp / (fp + tp + fn) if (fp + tp + fn) > 0 else 0  # False Alarm Rate
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity
        else:  # Multiclass classification
            tn = fp = fn = tp = fpr = far = specificity = None  # Not applicable
        
        accuracy = accuracy_score(y, y_pred)
        mcc = matthews_corrcoef(y, y_pred)  # Matthews Correlation Coefficient

        # AUC-ROC
        if len(self.classifier.classes_) == 2:
            y_proba = self.classifier.predict_proba(X_reduced)[:, 1]
            auc_roc = roc_auc_score(y, y_proba)
        else:
            auc_roc = None  # Not applicable for multiclass without binarization

        # Display results
        print("Prediction Results:")
        print("Classifier:", self.classifier)
        print("\nConfusion Matrix:\n\n", conf_matrix)
        print("\nClassification Report:\n\n", classification_report(y, y_pred, digits=6))

        #print("-" * 80)
        print("Metric Results:")
        print("-" * 80)
        print(f"{'Accuracy':<10}{'Time (s)':<10}{'FPR':<10}{'FAR':<10}{'Specificity':<15}{'AUC-ROC':<15}{'MCC':<15}")
        print("-" * 80)
        print(f"{accuracy:<10.6f}{inference_time:<10.4f}{(fpr if fpr is not None else '-'): <10}{(far if far is not None else '-'): <10}"
          f"{(specificity if specificity is not None else '-'): <15}{(auc_roc if auc_roc is not None else '-'): <15}{mcc:<15.6f}")
        print("-" * 80)
