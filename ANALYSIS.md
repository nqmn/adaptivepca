### Mathematical Analysis of AdaptivePCA

This analysis covers the mathematical framework, computational efficiency, and trade-offs of the `AdaptivePCA` algorithm as compared to a traditional `GridSearch` approach.

---

### 1. **Core Mathematical Function**

The `AdaptivePCA` algorithm's mathematical process is composed of scaling, PCA transformation, variance calculation, and optimal component selection.

#### Step-by-Step Breakdown

1. **Input Matrix $\( X \in \mathbb{R}^{n \times p} \)$**:
   - $\( n \)$: Number of samples
   - $\( p \)$: Number of features

2. **Step 1: Scaling Function $\( S(X) \)$**
   - The algorithm supports two scaling methods:
     - **StandardScaler**: Centers and normalizes data with the formula:
       $$\[
       S_1(X) = \frac{X - \mu}{\sigma}
       \]$$
       where $\( \mu \)$ is the mean and $\( \sigma \)$ the standard deviation of $\( X \)$.
     - **MinMaxScaler**: Scales data within a specified range, typically [0, 1], using:
       $$\[
       S_2(X) = \frac{X - X_{\min}}{X_{\max} - X_{\min}}
       \]$$

3. **Step 2: PCA Transformation**
   - After scaling, the algorithm applies PCA on the scaled data $\( X' = S(X) \)$ to reduce dimensionality.
   - **Objective**: Find the transformation matrix $\( W_k \)$ that maximizes the variance captured by $\( k \)$ components:
     $$\[
     W_k = \arg\max_W \{ \text{tr}(W^T X'^T X' W) \}
     \]$$
     subject to $\( W^T W = I_k \)$, where $\( W \in \mathbb{R}^{p \times k} \)$ and $\( k \leq \min(n, p) \)$.

4. **Step 3: Variance Ratio Function $\( V(k) \)$**
   - To select the number of components $\( k \)$, the cumulative variance explained by the first $\( k \)$ principal components is computed:
     $$\[
     V(k) = \frac{\sum_{i=1}^{k} \lambda_i}{\sum_{i=1}^{p} \lambda_i}
     $$\]
     where $\( \lambda_i \)$ are the eigenvalues of $\( X'^T X' \)$, representing the variance explained by each principal component.

5. **Step 4: Optimal Component Selection**
   - The smallest number of components $\( k^* \)$ that satisfies a specified variance threshold (e.g., 95%) is selected:
     $$\[
     k^* = \min \{ k : V(k) \geq \text{threshold} \}
     \]$$
   - The final `AdaptivePCA` function can be expressed as:
     $$\[
     F(X) = \arg\min_{s, k} \{ k : V_{s, k}(X) \geq \text{threshold} \}
     \]$$
   where $\( V_{s, k} \)$ is the variance ratio for scaler $\( s \)$ and $\( k \)$ components.

---

### 2. **Comparison with GridSearch PCA**

In a `GridSearch` approach, the algorithm exhaustively evaluates each scaler and a range of components, selecting the configuration that minimizes a loss function across the parameter space.

#### Key Mathematical Differences

1. **Search Strategy**
   - **AdaptivePCA**: Uses an early stopping criterion to select the minimum $\( k \)$ that meets the variance threshold.
   - **GridSearch**: Evaluates all combinations of scalers and component counts, choosing the combination with the lowest loss across the entire grid.

2. **Computational Complexity**
   - **AdaptivePCA**:
     $$\[
     O(np^2 + p^3) \times |S| \times k^*
     \]$$
     where:
     - $\( |S| \)$ is the number of scalers
     - $\( k^* \)$ is the optimal component count
   - **GridSearch**:
     $$\[
     O(np^2 + p^3) \times |S| \times |K|
     \]$$
     where $\( |K| \)$ is the total number of component values in the grid.

3. **Optimization Goal**
   - **AdaptivePCA**: Minimizes $\( k \)$ such that $\( V(k) \geq \text{threshold} \)$.
   - **GridSearch**: Minimizes the loss function $\( L(X, s, k) \)$ over the entire grid, which could achieve a global minimum but at a higher computational cost.

---

### 3. **Efficiency Analysis**

The efficiency of `AdaptivePCA` compared to `GridSearch` is measured by the ratio \( E \) between the number of configurations each approach evaluates:

$$\[
E = \frac{k^* \times |S|}{|K| \times |S|} = \frac{k^*}{|K|}
\]$$

where $\( k^* \leq |K| \)$. This implies that `AdaptivePCA` is more efficient when the variance threshold is met early, reducing the need to search through all values of $\( k \)$.

---

### 4. **Trade-offs**

1. **Optimality Guarantee**
   - **AdaptivePCA**: Provides a local optimum by selecting the first $\( k \)$ meeting the threshold, which may not necessarily be globally optimal.
   - **GridSearch**: Exhaustively searches the parameter space, thus yielding a global optimum for the specified grid but at a computational expense.

2. **Computational Efficiency**
   - **AdaptivePCA**:
     $$\[
     O(np^2 + p^3) \times |S| \times k^*
     \]$$
   - **GridSearch**:
     $$\[
     O(np^2 + p^3) \times |S| \times |K|
     \]$$
   The reduced search space in `AdaptivePCA` translates to a faster runtime when $\( k^* \)$ is significantly smaller than $\( |K| \)$.

3. **Memory Requirements**
   - **AdaptivePCA**: Requires $\( O(np + k^* p) \)$ space, as it only maintains the current best configuration up to $\( k^* \)$.
   - **GridSearch**: Requires $\( O(np + |K| p) \)$ memory, storing all configurations evaluated within the grid.

---

### Summary

- **AdaptivePCA**: Efficient for applications where meeting a specific variance threshold quickly is more important than exhaustively searching for a global minimum. It provides a locally optimal solution with lower computational cost and memory usage.
- **GridSearch PCA**: Suitable when a globally optimal configuration is required across all scalers and component counts, but it is computationally more expensive.

The adaptive nature of `AdaptivePCA` makes it well-suited for large datasets where dimensionality reduction speed is critical, while `GridSearch` offers a thorough exploration at a cost of time and memory.