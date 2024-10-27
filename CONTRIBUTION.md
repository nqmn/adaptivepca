# Key Mathematical Differences of AdaptivePCA

Below are the key mathematical differences that make `AdaptivePCA` novel compared to traditional PCA methods, such as `GridSearch PCA`.

## 1. Early Stopping Criterion
- **AdaptivePCA** uses an early stopping criterion based on a variance threshold $`V(k) \geq \text{threshold}`$. It selects the smallest number of components $`k`$ that meets this threshold.
- This allows **AdaptivePCA** to avoid unnecessary calculations for additional components once the target variance is reached, making it computationally efficient.
- In contrast, **GridSearch PCA** evaluates all possible component counts, leading to higher computational costs as it lacks an early termination criterion.

## 2. Dynamic Component Limit
- In **AdaptivePCA**, the maximum number of components $`k^*`$ is dynamically limited to $`\min(n, p)`$, where $`n`$ is the number of samples and $`p`$ is the number of features, ensuring only feasible dimensions are considered.
- This adaptability allows **AdaptivePCA** to efficiently handle datasets with varying shapes, unlike traditional methods that may have a fixed maximum component count.

## 3. Variance Ratio as Objective Function
- **AdaptivePCA** directly optimizes the cumulative variance ratio function $`V(k)`$ as the primary objective, selecting $`k`$ to achieve a target variance threshold.
- This contrasts with **GridSearch PCA**, which optimizes a more generic loss function $`L(X, s, k)`$ across a grid of scalers and components without a variance-specific focus.
- By using the variance ratio as an objective, **AdaptivePCA** ensures that dimensionality reduction is closely aligned with data variability, making it more targeted for capturing essential variance.

## 4. Scalable Component Selection with Adaptive Thresholding
- By adjusting the threshold based on cumulative variance, **AdaptivePCA** identifies the minimal number of components needed for the desired data representation, achieving a balance between dimensionality reduction and variance retention.
- **GridSearch PCA**, however, uses a predetermined and fixed number of components within a grid, which lacks the adaptive flexibility of **AdaptivePCA** to stop once the target variance is met.

## 5. Scaler-Specific Variance Ratio Calculation
- **AdaptivePCA** calculates the variance ratio $`V_{s,k}(X)`$ for each scaler $`s`$ independently, allowing it to assess how scaling affects variance distribution.
- This scaler-specific approach helps **AdaptivePCA** select the scaler that best enhances the data structure for PCA.
- In contrast, **GridSearch PCA** considers scalers but evaluates them across the entire grid without variance-specific optimization, making **AdaptivePCA** more tailored in its approach to scaling.

## 6. Reduced Computational Complexity
- The computational complexity of **AdaptivePCA** is $`O(np^2 + p^3) \times |S| \times k^*`$, where $`k^*`$ is the minimum number of components that meet the variance threshold.
- In comparison, **GridSearch PCA** has a complexity of $`O(np^2 + p^3) \times |S| \times |K|`$, where $`|K|`$ is the full set of components in the grid.
- When $`k^*`$ is smaller than $`|K|`$, **AdaptivePCA** achieves significant efficiency gains, particularly on large datasets.

## Summary of Novelty

The novel aspects of **AdaptivePCA** are:
- **Early stopping criterion** based on a target variance threshold.
- **Dynamic component limit** adaptable to the data’s shape.
- **Direct variance ratio optimization**, focusing on cumulative variance as the objective.
- **Scaler-specific optimization** of variance contribution, selecting the scaler that best enhances the data structure for PCA.
- **Reduced computational complexity**, making it efficient for large-scale datasets.

These mathematical innovations enable **AdaptivePCA** to balance efficiency and accuracy in dimensionality reduction, addressing computational overhead often seen in traditional methods like **GridSearch PCA**.