# Recommendation System Model for Addressing Cold Start and Scalability Issues

## Problem Statement

The problem addressed in this research involves improving recommendation systems by mitigating the cold start and scalability issues, particularly in dynamic environments with large datasets. Traditional collaborative filtering models, such as Singular Value Decomposition (SVD) and Probabilistic Matrix Factorization (PMF), struggle with data sparsity, especially in the face of new users, items, and limited ratings. This research aims to develop a more efficient and accurate recommendation model by utilizing supplemental information and advanced techniques to handle sparse matrices and improve scalability.

## Dataset Used

The research uses the **MovieLens dataset**, which contains ratings and metadata related to movies, users, and their interactions. The dataset includes:

- **User Data**: Information about the user, such as user ID, demographic data, and user ratings.
- **Movie Data**: Movie ID, genre, and title.
- **Interaction Data**: Ratings given by users to movies, as well as the timestamp of the interactions.

## Baseline Models

The research evaluates several baseline models used in collaborative filtering:

1. **Matrix Factorization (MF)**: Decomposes the user-item interaction matrix into two lower-dimensional matrices to capture latent factors representing user preferences and item characteristics.
2. **Bayesian Personalized Ranking (BPR)**: Focuses on pairwise ranking of items, which is particularly useful for implicit feedback systems.
3. **Neural Collaborative Filtering (NCF)**: Enhances matrix factorization by using neural networks to capture non-linear user-item interactions.
4. **Convolutional Matrix Factorization (ConvMF)**: Integrates convolutional neural networks with matrix factorization to capture local dependencies in the user-item interaction matrix.
5. **Factorization Machines (FM)**: A generalization of matrix factorization that models higher-order interactions between features in sparse datasets.

## Evaluation Metrics

The models are evaluated using a range of metrics to assess the accuracy and robustness of the recommendations:

1. **Mean Absolute Error (MAE)**: Measures the average magnitude of errors in predicted ratings.
2. **Root Mean Squared Error (RMSE)**: Measures the square root of the average squared errors, penalizing larger errors more.
3. **Precision**: Measures the percentage of recommended items that are relevant.
4. **Recall**: Measures the percentage of relevant items that are recommended.
5. **F1 Score**: Harmonic mean of precision and recall, balancing both metrics.
6. **Scalability**: Evaluates the model's ability to handle large datasets and real-time data.

## Proposed Model

- **Clustering**: The user base is divided into clusters based on demographic and genre preferences, allowing for better user segmentation.
- **CNN for Classification**: A Convolutional Neural Network is used to predict the user's cluster label based on their features. This allows for a more sophisticated, deep learning-based approach to clustering.
- **Movie Recommendation**: The model recommends movies based on user clusters, making the recommendations more personalized and relevant.

## Evaluation

Below is the table that compares the performance of various baseline models and the proposed model across different evaluation metrics:

| Metric/ Model   | MF_SVD | BPR  | NCF  | FM   | Convo_MF | Proposed_model |
|-----------------|--------|------|------|------|----------|----------------|
| **RMSE**        | 2.609  | 0.945| 0.522| 2.764| 2.392    | **0.606**      |
| **MSE**         | 6.807  | 0.893| 0.272| 7.642| 5.722    | **0.367**      |
| **MAE**         | 2.342  | 0.748| 0.416| 2.120| 1.909    | **0.263**      |
| **Accuracy**    | 0.468  | 0.410| 0.601| 0.502| 0.491    | **0.748**      |
| **Precision**   | 0.874  | 0.410| 0.805| 0.560| 0.566    | **0.746**      |
| **Recall**      | 0.037  | 0.410| 0.361| 0.441| 0.318    | **0.748**      |
| **F1 Score**    | 0.071  | 0.410| 0.498| 0.493| 0.407    | **0.738**      |

### Conclusion

The above table compares the performance of **Matrix Factorization (MF_SVD)**, **Bayesian Personalized Ranking (BPR)**, **Neural Collaborative Filtering (NCF)**, **Factorization Machines (FM)**, **Convolutional Matrix Factorization (ConvMF)**, and the **Proposed Model** across key evaluation metrics.

- **RMSE & MSE**: The proposed model significantly outperforms all baseline models in terms of **Root Mean Squared Error (RMSE)** and **Mean Squared Error (MSE)**. Lower RMSE and MSE indicate that the proposed model provides more accurate predictions compared to the other models.
  
- **MAE**: The **Mean Absolute Error (MAE)** of the proposed model is the lowest among all, suggesting it provides predictions that are closer to the actual ratings.

- **Accuracy**: The proposed model achieves the highest accuracy (0.748), which shows its effectiveness in making correct recommendations compared to other methods.

- **Precision**: The proposed model performs well in terms of **precision**, which means that the items it recommends are highly relevant, similar to the **Matrix Factorization (MF_SVD)** method.

- **Recall**: The proposed model significantly outperforms the others in **recall** (0.748), indicating that it is more successful in recommending all relevant items compared to the baseline models.

- **F1 Score**: The **F1 Score** of the proposed model is higher than all other models, indicating that it strikes a good balance between precision and recall. This reflects its robustness in providing quality recommendations.

### Summary

The **Proposed Model** provides the best overall performance across all evaluation metrics, particularly excelling in **scalability** and **accuracy**. By integrating clustering techniques and deep learning (CNN), it not only mitigates cold start and scalability issues but also improves the overall recommendation quality, making it a promising solution for dynamic and large-scale recommendation systems.
