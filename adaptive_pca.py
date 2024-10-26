import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import pandas as pd
from typing import Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

class AdaptivePCA:
    def __init__(self, variance_threshold: float = 0.95, max_components: int = 10):
        self.variance_threshold = variance_threshold
        self.max_components = max_components
        self.best_scaler = None
        self.best_n_components = None
        self.scalers = {
            'StandardScaler': StandardScaler(),
            'MinMaxScaler': MinMaxScaler()
        }
        self.results = []  # Store results for each approach

    def _apply_scaler(self, X: pd.DataFrame, scaler) -> np.ndarray:
        return scaler.fit_transform(X)
    
    def _evaluate_pca(self, X_scaled: np.ndarray, scaler_name: str) -> Optional[Dict[str, float]]:
        pca = PCA(n_components=min(self.max_components, X_scaled.shape[1]))
        cumulative_variance = np.cumsum(pca.fit(X_scaled).explained_variance_ratio_)

        for n_components in range(1, self.max_components + 1):
            explained_variance_score = cumulative_variance[n_components - 1]
            self.results.append({
                'Scaler': scaler_name,
                'Components': n_components,
                'Score': explained_variance_score,
            })
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
        print("-" * 50)
        print(f"{'Scaler':<20}{'Components':<12}{'Score':<12}")
        print("-" * 50)
        start_time = time.time()
        
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self._evaluate_scaler_and_pca, X, scaler_name, scaler): scaler_name 
                for scaler_name, scaler in self.scalers.items()
            }
            
            for future in as_completed(futures):
                best_config = future.result()
                if best_config:
                    self.best_scaler = best_config['best_scaler']
                    self.best_n_components = best_config['best_n_components']
                    self.best_explained_variance = best_config['best_explained_variance']
                    break

        self._display_final_results(time.time() - start_time)

    def _evaluate_scaler_and_pca(self, X: pd.DataFrame, scaler_name: str, scaler) -> Optional[Dict[str, float]]:
        X_scaled = self._apply_scaler(X, scaler)
        return self._evaluate_pca(X_scaled, scaler_name)

    def _display_final_results(self, elapsed_time: float):
        if self.best_scaler and self.best_n_components:
            print("\nBest configuration found:")
            print("-" * 70)
            print(f"{'Best Scaler':<20}{'Optimal Components':<20}{'Best Score':<15}{'Time (s)':<15}")
            print("-" * 70)
            print(f"{self.best_scaler:<20}{self.best_n_components:<20}{self.best_explained_variance:<15.6f}{elapsed_time:.4f}")
            print("-" * 70)
        else:
            print("\nNo configuration met the variance threshold.")

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self.best_scaler or not self.best_n_components:
            raise RuntimeError("You must fit the AdaptivePCA model before calling transform.")
        
        scaler = self.scalers[self.best_scaler]
        X_scaled = self._apply_scaler(X, scaler)
        pca = PCA(n_components=self.best_n_components)
        return pca.fit_transform(X_scaled)

    def fit_transform(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        self.fit(X)
        return self.transform(X) if self.best_scaler and self.best_n_components else None
