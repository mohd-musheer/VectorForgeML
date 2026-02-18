module.exports = {
    // ==========================================
    // MODELS
    // ==========================================
    "DecisionTree": {
        title: "Decision Tree Regressor",
        description: "A high-performance, recursive partitioning algorithm that constructs a regression tree by exhaustively searching for optimal splits. Unlike standard implementations, vectorForgeML's Decision Tree is built directly in C++ using raw pointers for graph construction, ensuring minimal memory overhead and cache-friendly traversal. It supports deep trees and handles continuous features with high precision. The model predicts the target value by traversing the tree from root to leaf, where each node represents a decision rule based on a single feature threshold.",
        steps: [
            "START: Initialize the tree building process with the full dataset (X, y) at the root node.",
            "CHECK: Verify if current depth < max_depth and if variance in target y > 0.",
            "SEARCH: Iterate through ALL features (j=1..p) and ALL unique thresholds to find the optimal split.",
            "CRITERION: Calculate the Cost Function (MSE Reduction) for every potential split: Cost = (n_L/n)*MSE_L + (n_R/n)*MSE_R.",
            "SPLIT: Partition the data into Left (L) and Right (R) subsets based on the best feature/threshold pair.",
            "RECURSE: Recursively call the build function on L and R subsets.",
            "LEAF: Create a leaf node if stopping criteria are met. Store the mean value $\\bar{y}$ of the node's samples.",
            "OPTIMIZE: Use index-based views (no deep copies) to pass data subsets to child nodes.",
            "END: Return the root pointer of the constructed tree structure."
        ],
        math: "$$ \\min_{j, s} \\left[ \\min_{c_1} \\sum_{x_i \\in R_1(j,s)} (y_i - c_1)^2 + \\min_{c_2} \\sum_{x_i \\in R_2(j,s)} (y_i - c_2)^2 \\right] $$",
        impl: "Implemented in `DecisionTree.cpp`. Recursively partitions data.",
        code: `Node* build(NumericMatrix X, NumericVector y, int depth,int maxd){
    // ... (find best split based on minimizing MSE) ...
    if(depth>=maxd || bf==-1){
        node->value=s/n; // Leaf node: store mean
        return node;
    }
    node->left=build(XL,yL,depth+1,maxd);
    node->right=build(XR,yR,depth+1,maxd);
    return node;
}`,
        time: "Training: O(N * P * D). Prediction: O(D).",
        space: "O(Nodes).",
        opt: "Recursive implementation with index passing.",
        cases: "Non-linear regression.",
        limitations: "Regression only."
    },
    "RandomForest": {
        title: "Random Forest",
        description: "A robust ensemble meta-estimator that fits a number of decision tree classifiers/regressors on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. Our implementation utilizes `std::vector` and raw pointers for managing a forest of trees, with specific optimizations for random feature selection at each node split.",
        steps: [
            "START: Initialize the forest with `ntrees` empty trees.",
            "BOOTSTRAP: For each tree t=1..T, generate a random sample of size N from X with replacement.",
            "BUILD: Grow a decision tree on the bootstrapped data.",
            "FEATURE SUBSET: At each node, select a random subset of features of size `mtry`.",
            "BEST SPLIT: Find the best split among the `mtry` features only.",
            "GROW: Continue splitting until max_depth is reached (no pruning).",
            "STORE: Save the root pointer of the tree in the forest container.",
            "PREDICT: For a new sample, traverse all T trees.",
            "AGGREGATE: Compute the average (Regression) or Majority Vote (Classification) of all tree predictions.",
            "END: Return the final ensemble prediction."
        ],
        math: "$$ \\hat{y}(x) = \\frac{1}{T} \\sum_{t=1}^T f_t(x) $$",
        impl: "Implemented in `RandomForest.cpp`.",
        code: `for(int i=0; i<ntrees; ++i) {
    // Bootstrap sampling
    // Random feature selection
    trees[i] = build_tree(bootstrap_X, bootstrap_y, mtry);
}
// Prediction
for(Node* t : trees)
    preds += predict_tree(t, X);
preds /= ntrees;`,
        time: "Training: O(T * N * log(N) * mtry).",
        space: "O(T * Nodes).",
        opt: "Ensemble variance reduction.",
        cases: "Robust classification/regression.",
        limitations: "Single-threaded build."
    },
    "LinearRegression": {
        title: "Linear Regression",
        description: "A fundamental statistical method that models the relationship between a scalar response and one or more explanatory variables using a linear predictor functions. This implementation solves the Ordinary Least Squares (OLS) problem using the Normal Equation approach. It is heavily optimized using BLAS (Basic Linear Algebra Subprograms) and LAPACK (Linear Algebra PACKage) for native hardware acceleration, making it suitable for high-performance computing tasks.",
        steps: [
            "STARTINPUT: Matrix $X$ (Features) and Vector $y$ (Targets).",
            "VALIDATE: Check for row/column consistency and dimensions.",
            "PARALLEL MEAN: Compute $\\bar{x}$ for each column via OpenMP SIMD reduction.",
            "GRAM MATRIX: Calculate $X^TX$ using vectorized BLAS `dgemm` (Matrix-Matrix Mutliplication).",
            "TARGET PROJECTION: Calculate $X^Ty$ using BLAS `dgemv` (Matrix-Vector Multiplication).",
            "CENTERING: Adjust $X^TX$ and $X^Ty$ using $\\bar{x}$ to effectively center the data.",
            "STABILIZE: Add a small $10^{-8}$ Ridge penalty to the diagonal for numerical stability.",
            "SOLVE: Apply LAPACK `dposv` (Cholesky Decomposition) to solve $A\\beta = b$.",
            "RECOVERY: If Cholesky fails (singular matrix), trigger Custom Fallback Solver (LU/QR).",
            "INTERCEPT: Compute $\\beta_0 = \\bar{y} - \\sum (\\beta_j \\cdot \\bar{x}_j)$ derived from means.",
            "OUTPUT: Store coefficients $\\beta$ and intercept $\\beta_0$.",
            "PREDICT: Compute standard dot product $X_{new} \\cdot \\beta + \\beta_0$."
        ],
        math: "$$ \\hat{\\beta} = (X^T X)^{-1} X^T y $$",
        impl: "Implemented in `LinearRegression.cpp` using BLAS/LAPACK.",
        code: `// Compute XtX and Xty using BLAS dgemm/dgemv
F77_CALL(dgemm)("T", "N", &p, &p, &n, &one, x, &n, x, ...);
// Solve using Cholesky (dposv)
F77_CALL(dposv)("L", &n, &one, XtX, &lda, Xty, ...);`,
        time: "O(N * P^2).",
        space: "O(P^2).",
        opt: "BLAS Level 3.",
        cases: "Baseline regression.",
        limitations: "Assumes linearity."
    },
    "LogisticRegression": {
        title: "Logistic Regression",
        description: "A generalized linear model used for binary classification. It estimates the parameters of a logistic model using iterative optimization. Our implementation uses a raw C++ full-batch Gradient Descent optimizer with an inlined sigmoid activation function for maximum throughput. It handles large datasets by processing the entire batch in memory (or chunks if extended) and updates weights based on the gradient of the Log-Likelihood.",
        steps: [
            "START: Initialize weight vector $\\mathbf{w}$ and bias $b$ to zeros.",
            "EPOCH LOOP: Iterate for a fixed set of `epochs`.",
            "LINEAR: Compute linear response $z_i = \\mathbf{w}^T \\mathbf{x}_i + b$ for all samples.",
            "ACTIVATE: Apply Sigmoid function $p_i = \\frac{1}{1 + e^{-z_i}}$.",
            "ERROR: Compute prediction error $e_i = p_i - y_i$.",
            "GRADIENT: Calculate $\\nabla w = X^T (\\hat{y} - y)$ accumulating over all samples.",
            "UPDATE: Apply update rule $\\mathbf{w} \\leftarrow \\mathbf{w} - \\eta \\nabla w$ (where $\\eta$ is learning rate).",
            "CONVERGE: Repeat until max epochs.",
            "PREDICT: Return 1 if $p_i > 0.5$ else 0."
        ],
        math: "$$ J(\\theta) = -\\frac{1}{m} \\sum_{i=1}^m [y^{(i)}\\log(h_\\theta(x^{(i)})) + (1-y^{(i)})\\log(1-h_\\theta(x^{(i)}))] $$",
        impl: "Implemented in `LogisticRegression.cpp`.",
        code: `for(int e=0; e<epochs; e++){
    double z = dot_product(row, coef) + intercept;
    double p = 1.0 / (1.0 + exp(-z));
    // Gradient update
    coef += learning_rate * (y - p) * row;
}`,
        time: "O(Epochs * N * P).",
        space: "O(P).",
        opt: "Sigmoid inline.",
        cases: "Binary classification.",
        limitations: "Linear boundary."
    },
    "SoftmaxRegression": {
        title: "Softmax Regression",
        description: "An extension of Logistic Regression to multi-class problems. It models the probability that a sample belongs to a specific class k using the Softmax function. The implementation relies on the 'Log-Sum-Exp' trick to prevent numerical instability (overflow/underflow) during the exponentiation step. This is critical for stable training on datasets with unscaled features.",
        steps: [
            "START: Identify $K$ unique classes and initialize $W$ matrix ($P \\times K$).",
            "EPOCH LOOP: Iterate through optimization steps.",
            "LOGITS: Compute raw scores $Z = XW$.",
            "MAX TRICK: Find $M = \\max(Z)$ for numerical stability.",
            "PROBABILITIES: Compute Softmax $P(k|x) = \\exp(Z_k - M) / \\sum \\exp(Z_j - M)$.",
            "LOSS GRAD: Compute gradients $\\nabla W$ based on Cross-Entropy Loss.",
            "UPDATE: Adjust weights $W \\leftarrow W - \\eta \\nabla W$.",
            "END: Return trained weights for $K$ classes."
        ],
        math: "$$ P(y=j|\\mathbf{x}) = \\frac{e^{\\mathbf{w}_j^T \\mathbf{x}}}{\\sum_{k=1}^K e^{\\mathbf{w}_k^T \\mathbf{x}}} $$",
        impl: "Implemented in `SoftmaxRegression.cpp`.",
        code: `// Softmax with Log-Sum-Exp trick
double max_z = max(logits);
double sum_exp = 0;
for(double z : logits) sum_exp += exp(z - max_z);
probs[k] = exp(logits[k] - max_z) / sum_exp;`,
        time: "O(Epochs * N * P * K).",
        space: "O(P * K).",
        opt: "Log-Sum-Exp stability.",
        cases: "Multiclass classification.",
        limitations: "Linear boundaries."
    },
    "RidgeRegression": {
        title: "Ridge Regression",
        description: "A regularized linear regression method that solves the ill-posed problem where features are highly correlated (multicollinearity). By adding an L2 penalty to the loss function, it shrinks the coefficients towards zero, reducing model variance. Our implementation efficiently adds this penalty directly to the Gram matrix ($X^T X$) before solving, utilizing the same optimized BLAS routines as standard Linear Regression.",
        steps: [
            "START: Receive Matrix X and Vector y.",
            "COMPUTE: Calculate the covariance/Gram matrix $A = X^T X$.",
            "REGULARIZE: Add $\\lambda$ (alpha) to the diagonal elements: $A_{ii} \\leftarrow A_{ii} + \\lambda$.",
            "RHS: Calculate the right-hand side vector $b = X^T y$.",
            "SOLVE: Solve the linear system $A \\beta = b$ using Cholesky Decomposition.",
            "FALLBACK: If Cholesky fails (rare), use LU Decomposition.",
            "END: Return robust coefficients $\\beta$."
        ],
        math: "$$ \\hat{\\beta}^{ridge} = (X^T X + \\lambda I)^{-1} X^T y $$",
        impl: "Implemented in `RidgeRegression.cpp`.",
        code: `// Add regularization
for(int j=0; j<p; ++j)
    XtX[j + j*p] += alpha;
// Solve
F77_CALL(dposv)(...);`,
        time: "O(N * P^2).",
        space: "O(P^2).",
        opt: "Cholesky Solver.",
        cases: "Correlated features.",
        limitations: "No feature selection."
    },
    "KNN": {
        title: "K-Nearest Neighbors",
        description: "A non-parametric, lazy learning algorithm that does not 'learn' a fixed model but instead stores the training instances. Prediction is performed by searching for the k most similar instances in the feature space. Our implementation is optimized for dense vectors using efficient Euclidean distance calculations and C++ standard library partial sorting to find the top-k neighbors without sorting the entire dataset.",
        steps: [
            "TRAINING: Store the entire dataset $X_{train}$ and $y_{train}$ in memory.",
            "QUERY: For each new sample $x_{q}$ to predict:",
            "DISTANCE: Compute Squared Euclidean Distance $d(x_q, x_i)$ for all $i=1..N$.",
            "SEARCH: Use `std::partial_sort` to find the $k$ smallest distances in $O(N \\log K)$ time.",
            "RETRIEVE: Get the labels of these $k$ nearest neighbors.",
            "VOTE: For Classification, assign the most frequent class (Mode).",
            "AVERAGE: For Regression, assign the mean value of the neighbors.",
            "END: Return the predicted value."
        ],
        math: "$$ d(\\mathbf{p}, \\mathbf{q}) = \\sqrt{ \\sum_{i=1}^n (q_i - p_i)^2 } $$",
        impl: "Implemented in `KNN.cpp`.",
        code: `// Partial sort to find top-k
std::partial_sort(indices.begin(), indices.begin()+k, indices.end(),
    [&](int i, int j){ return dists[i] < dists[j]; });
// Vote
for(int i=0; i<k; ++i) votes[y[indices[i]]]++;`,
        time: "Pred: O(N * P).",
        space: "O(N * P).",
        opt: "Partial Sort.",
        cases: "Pattern recognition.",
        limitations: "Slow prediction."
    },
    "KMeans": {
        title: "K-Means Clustering",
        description: "An iterative unsupervised learning algorithm that partitions a dataset into K distinct, non-overlapping subgroups (clusters). Each data point belongs to the cluster with the nearest mean. The algorithm minimizes the within-cluster sum of squares (WCSS). It employs the classic Lloyd's algorithm with efficient C++ loops for distance calculation and centroid updates.",
        steps: [
            "START: Specify $K$ clusters.",
            "INIT: Randomly select $K$ data points as initial centroids $\\mu^{(0)}$.",
            "ASSIGN: For every point $x_i$, find the nearest centroid $c_j$ minimizing $||x_i - \\mu_j||^2$.",
            "UPDATE: Recalculate each centroid $\\mu_j$ as the mean of all points assigned to it.",
            "CHECK: If centroids have not changed (or change < epsilon), STOP.",
            "LOOP: Otherwise, repeat the Assign and Update steps.",
            "END: Return cluster labels and final centroid locations."
        ],
        math: "$$ \\text{arg} \\min_S \\sum_{i=1}^k \\sum_{\\mathbf{x} \\in S_i} ||\\mathbf{x} - \\mathbf{\\mu}_i||^2 $$",
        impl: "Implemented in `KMeans.cpp`.",
        code: `while(changed && iter < max_iter) {
    // Assignment
    int c = nearest_centroid(row, centroids);
    // Update
    new_centroids[c] += row;
    counts[c]++;
}`,
        time: "O(I * N * K * P).",
        space: "O(K * P).",
        opt: "In-place updates.",
        cases: "Clustering.",
        limitations: "Local optima."
    },
    "PCA": {
        title: "Principal Component Analysis",
        description: "A statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables called principal components. Our implementation focuses on numerical accuracy using the Covariance method and LAPACK's symmetric eigenvalue solver.",
        steps: [
            "START: Input data matrix $X$.",
            "CENTER: Subtract the mean of each column from the data.",
            "COVARIANCE: Compute the covariance matrix $C = \\frac{1}{N-1} X^T X$.",
            "EIGEN: Solving the eigenvalue problem $C v = \\lambda v$ using LAPACK `dsyev`.",
            "SORT: Order eigenvalues $\\lambda$ descending and reorder eigenvectors $v$.",
            "SELECT: Choose the top $k$ eigenvectors corresponding to the largest eigenvalues.",
            "TRANSFORM: Project $X$ onto the new subspace: $X_{new} = X \\cdot V_k$.",
            "END: Return the transformed data and components."
        ],
        math: "$$ X = T P^T + E $$",
        impl: "Implemented in `PCA.cpp` using LAPACK `dsyev` (Symmetric Eigen Value solver). This is numerically robust. The Covariance matrix computation is manual C++ loops.",
        code: `// Compute Covariance
// ...
// Eigen decomposition
F77_CALL(dsyev)("V", "U", &p, cov.data(), ...);
// Select top k vectors`,
        time: "O(P^3).",
        space: "O(P^2).",
        opt: "LAPACK dsyev.",
        cases: "Dimensionality reduction.",
        limitations: "Linear."
    },

    // ==========================================
    // PREPROCESSING
    // ==========================================
    // ==========================================
    // PREPROCESSING
    // ==========================================
    // ==========================================
    // PREPROCESSING
    // ==========================================
    "StandardScaler": {
        title: "Standard Scaler",
        description: "Standardizes features by removing the mean and scaling to unit variance.",
        impl: "Implemented in `scalers.R`. Uses `cpp_scale_fit_transform` for speed if available, otherwise R `scale`.",
        code: `fit=function(X){
  X <- as.matrix(X)
  if (exists("cpp_scale_fit_transform", mode="function")) {
    out <- cpp_scale_fit_transform(X)
    mean <<- out$mean
    sd <<- out$sd
  } else {
    mean <<- colMeans(X)
    sd <<- apply(X,2,sd)
    sd[sd==0] <<- 1
  }
}`,
        lang: "r",
        time: "O(N * P).",
        space: "O(P)."
    },
    "MinMaxScaler": {
        title: "Min-Max Scaler",
        description: "Transforms features by scaling each feature to a given range.",
        impl: "Implemented in `scalers.R`.",
        code: `transform=function(X){
  X <- as.matrix(X)
  (X - minv)/(maxv - minv + 1e-8)
}`,
        lang: "r",
        time: "O(N * P).",
        space: "O(P)."
    },

    // ==========================================
    // METRICS
    // ==========================================
    "accuracy_score": {
        title: "Accuracy Score",
        description: "Accuracy classification score.",
        impl: "Implemented in `metrics.R`.",
        code: `accuracy_score <- function(y_true, y_pred){
  y_true <- as.vector(y_true)
  y_pred <- as.vector(y_pred)
  mean(y_true == y_pred, na.rm = TRUE)
}`,
        lang: "r"
    },
    "confusion_matrix": {
        title: "Confusion Matrix",
        description: "Computes the confusion matrix.",
        impl: "Implemented in `metrics.R`.",
        code: `confusion_matrix <- function(y_true, y_pred){
  classes <- sort(unique(c(y_true,y_pred)))
  k <- length(classes)
  mat <- matrix(0L,k,k, dimnames=list(Actual=classes, Predicted=classes))
  for(i in seq_along(y_true)){
    a <- match(y_true[i],classes)
    p <- match(y_pred[i],classes)
    mat[a,p] <- mat[a,p] + 1L
  }
  mat
}`,
        lang: "r"
    },
    "f1_score": {
        title: "F1 Score",
        description: "The harmonic mean of precision and recall.",
        impl: "Implemented in `metrics.R`.",
        code: `f1_score <- function(y_true, y_pred, positive = NULL){
  p <- precision_score(y_true, y_pred, positive)
  r <- recall_score(y_true, y_pred, positive)
  if(p+r == 0) return(0)
  2*p*r/(p+r)
}`,
        lang: "r"
    },

    // ==========================================
    // REMAINING PREPROCESSING
    // ==========================================
    "LabelEncoder": {
        title: "Label Encoder",
        description: "Encodes target labels with value between 0 and n_classes-1.",
        impl: "Implemented in `encoders.R`.",
        code: `fit=function(x){
  vals <- unique(x)
  map <<- setNames(seq_along(vals)-1, vals)
},
transform=function(x){
  as.numeric(map[x])
}`,
        lang: "r"
    },
    "OneHotEncoder": {
        title: "One Hot Encoder",
        description: "Encodes categorical features as a one-hot numeric array.",
        impl: "Implemented in `encoders.R`.",
        code: `transform=function(df){
  # ... setup code ...
  for(colname in names(df)){
    col <- as.character(df[[colname]])
    cats <- categories[[colname]]
    idx <- match(col, cats)
    # ... fill matrix ...
    out[cbind(row_ids, col_ids)] <- 1L
  }
  out
}`,
        lang: "r"
    },
    "drop_constant_columns": {
        title: "Drop Constant Columns",
        description: " Removes columns that have zero variance.",
        impl: "Implemented in C++ (`LinearRegression.cpp`).",
        code: `// Check variance
if(var < eps) {
    dropped_cols.push_back(j);
} else {
    keep_cols.push_back(j);
    // Copy column
}`
    },
    "ColumnTransformer": {
        title: "Column Transformer",
        description: "Applies transformers to columns of an array or pandas DataFrame.",
        impl: "Implemented in `column_transformer.R`.",
        code: `transform=function(df){
  # ...
  if(!is.null(num_pipeline) && length(num_cols) > 0L){
    parts[[length(parts)+1]] <-
      as_feature_matrix(num_pipeline$transform(df[,num_cols,drop=FALSE]))
  }
  # ... same for cat_pipeline ...
  out <- do.call(cbind, parts)
  out
}`,
        lang: "r"
    },
    "Pipeline": {
        title: "Pipeline",
        description: "Sequentially apply a list of transforms and a final estimator.",
        steps: [
            "Iterate transformer steps: fit_transform()",
            "Final step: fit()"
        ],
        impl: "Implemented in `pipeline.R`.",
        code: `fit=function(X,y){
  data <- X
  for(i in seq_len(n_steps)){
    step <- steps[[i]]
    if(pipeline_is_transformer(step)){
       if(pipeline_has_method(step,"fit_transform")){
         data <- step$fit_transform(data)
       } # ...
    } else if(pipeline_is_estimator(step)){
       step$fit(data,y)
    }
  }
}`,
        lang: "r"
    },
    "train_test_split": {
        title: "Train Test Split",
        description: "Split arrays or matrices into random train and test subsets.",
        impl: "Implemented in `split.R`.",
        code: `train_test_split <- function(X,y,test_size=0.2, seed=NULL){
  if(!is.null(seed)) set.seed(seed)
  n <- nrow(X)
  idx <- sample(n)
  split <- max(2, floor(n*(1-test_size)))
  train_idx <- idx[1:split]
  test_idx  <- idx[(split+1):n]
  list(X_train=X[train_idx,,drop=FALSE], X_test=X[test_idx,,drop=FALSE], 
       y_train=y[train_idx], y_test=y[test_idx])
}`,
        lang: "r"
    },

    // ==========================================
    // REMAINING METRICS
    // ==========================================
    "precision_score": {
        title: "Precision Score",
        description: "The ratio tp / (tp + fp).",
        impl: "Implemented in `metrics.R`.",
        code: `precision_score <- function(y_true, y_pred, positive = NULL){
  if(is.null(positive)) positive <- unique(y_true)[1]
  tp <- sum(y_true == positive & y_pred == positive)
  fp <- sum(y_true != positive & y_pred == positive)
  if(tp + fp == 0) return(0)
  tp/(tp+fp)
}`,
        lang: "r"
    },
    "recall_score": {
        title: "Recall Score",
        description: "The ratio tp / (tp + fn).",
        impl: "Implemented in `metrics.R`.",
        code: `recall_score <- function(y_true, y_pred, positive = NULL){
  if(is.null(positive)) positive <- unique(y_true)[1]
  tp <- sum(y_true == positive & y_pred == positive)
  fn <- sum(y_true == positive & y_pred != positive)
  if(tp + fn == 0) return(0)
  tp/(tp+fn)
}`,
        lang: "r"
    },
    "mse": {
        title: "Mean Squared Error",
        description: "Mean squared error regression loss.",
        impl: "Implemented in `metrics.R`.",
        code: `mse <- function(y_true,y_pred){
  mean((y_true-y_pred)^2, na.rm=TRUE)
}`,
        lang: "r"
    },
    "rmse": {
        title: "Root Mean Squared Error",
        description: "The square root of the mean squared error.",
        impl: "Implemented in `metrics.R`.",
        code: `rmse <- function(y_true,y_pred){
  sqrt(mse(y_true,y_pred))
}`,
        lang: "r"
    },
    "r2_score": {
        title: "R2 Score",
        description: "$R^2$ (coefficient of determination).",
        impl: "Implemented in `metrics.R`.",
        code: `r2_score <- function(y_true,y_pred){
  y_true <- y_true[!is.na(y_pred)]
  y_pred <- y_pred[!is.na(y_pred)]
  ss_res <- sum((y_true-y_pred)^2)
  ss_tot <- sum((y_true-mean(y_true))^2)
  if(is.na(ss_tot) || ss_tot==0) return(1)
  1 - ss_res/ss_tot
}`,
        lang: "r"
    },
    "macro_f1": {
        title: "Macro F1 Score",
        description: "Calculate F1 score for each class and find their unweighted mean.",
        impl: "Implemented in `metrics.R`.",
        code: `macro_f1 <- function(y_true, y_pred){
  classes <- unique(y_true)
  mean(sapply(classes, function(cls)
    f1_score(y_true, y_pred, cls)))
}`,
        lang: "r"
    },
    "confusion_stats": {
        title: "Confusion Stats",
        description: "summary of metrics from Conf Matrix.",
        impl: "Implemented in `metrics.R`.",
        code: `confusion_stats <- function(cm){
  total <- sum(cm)
  acc <- sum(diag(cm))/total
  precision <- diag(cm)/colSums(cm)
  recall <- diag(cm)/rowSums(cm)
  f1 <- 2*precision*recall/(precision+recall)
  list(accuracy = acc, precision = precision, recall = recall, f1 = f1)
}`,
        lang: "r"
    },

    // ==========================================
    // CORE UTILITIES
    // ==========================================
    "matrix_ops": {
        title: "Matrix Operations",
        description: "High-performance matrix utilities.",
        impl: "Implemented in `matrix_ops.cpp`.",
        code: `// Parallel Matrix Multiplication
#pragma omp parallel for
for(int i=0; i<n; i++){
    for(int j=0; j<p; j++){
        // dot product row i, col j
    }
}`,
        opt: "OpenMP."
    },
    "dot_product": {
        title: "Dot Product",
        description: "Computes the dot product of two vectors using BLAS Level 1 operations.",
        math: "$\\mathbf{a} \\cdot \\mathbf{b} = \\sum a_i b_i$",
        impl: "Implemented in `functions.cpp`.",
        code: `F77_CALL(ddot)(&n, a.begin(), &inc, b.begin(), &inc)`
    },
    "square_vec": {
        title: "Square Vector",
        description: "Computes the element-wise square of a vector.",
        math: "$y_i = x_i^2$",
        impl: "Implemented in `functions.cpp`.",
        code: `x * x  // Rcpp vectorized`
    },
    "cpp_sum_squares": {
        title: "Sum of Squares (Analytical)",
        description: "Computes the sum of squared integers.",
        math: "$\\sum_{i=1}^n i^2 = \\frac{n(n+1)(2n+1)}{6}$",
        impl: "Implemented in `functions.cpp`.",
        code: `return dn * (dn + 1.0) * (2.0 * dn + 1.0) / 6.0;`
    }
};
