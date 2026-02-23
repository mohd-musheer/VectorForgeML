import os
import re

R_DIR = "R"

def escape_latex(text):
    return text.replace('%', '\\%').replace('#', '\\#').replace('_', '\\_')

EXAMPLES = {
    "KMeans": "  x <- matrix(rnorm(20), nrow=10)\n  model <- KMeans$new()\n  model$fit(x)",
    "LinearRegression": "  model <- LinearRegression$new()\n  X <- matrix(rnorm(20), nrow=10)\n  y <- rnorm(10)\n  model$fit(X,y)\n  model$predict(X)",
    "DecisionTree": "  model <- DecisionTree$new()\n  X <- matrix(rnorm(20), nrow=10)\n  y <- sample(0:1, 10, replace=TRUE)\n  model$fit(X,y)\n  model$predict(X)",
    "KNN": "  model <- KNN$new(k=3, mode=\"classification\")\n  X <- matrix(rnorm(20), nrow=10)\n  y <- sample(0:1, 10, replace=TRUE)\n  model$fit(X,y)\n  model$predict(X)",
    "LogisticRegression": "  model <- LogisticRegression$new()\n  X <- matrix(rnorm(20), nrow=10)\n  y <- sample(0:1, 10, replace=TRUE)\n  model$fit(X,y)\n  model$predict(X)",
    "PCA": "  model <- PCA$new(n_components=2)\n  X <- matrix(rnorm(30), nrow=10)\n  model$fit(X)\n  model$transform(X)",
    "RandomForest": "  model <- RandomForest$new(n_trees=5)\n  X <- matrix(rnorm(20), nrow=10)\n  y <- sample(0:1, 10, replace=TRUE)\n  model$fit(X,y)\n  model$predict(X)",
    "RidgeRegression": "  model <- RidgeRegression$new(alpha=1.0)\n  X <- matrix(rnorm(20), nrow=10)\n  y <- rnorm(10)\n  model$fit(X,y)\n  model$predict(X)",
    "SoftmaxRegression": "  model <- SoftmaxRegression$new()\n  X <- matrix(rnorm(20), nrow=10)\n  y <- sample(0:2, 10, replace=TRUE)\n  model$fit(X,y)\n  model$predict(X)",
    "ColumnTransformer": "  model <- ColumnTransformer$new(num_cols=\"A\", cat_cols=\"B\")",
    "Pipeline": "  model <- Pipeline$new(list(StandardScaler$new()))",
    "LabelEncoder": "  enc <- LabelEncoder$new()\n  x <- c(\"a\", \"b\", \"a\")\n  enc$fit(x)\n  enc$transform(x)",
    "OneHotEncoder": "  enc <- OneHotEncoder$new()\n  df <- data.frame(a=c(\"x\",\"y\",\"x\"))\n  enc$fit(df)\n  enc$transform(df)",
    "confusion_matrix": "  y_true <- c(1,0,1,1)\n  y_pred <- c(1,0,0,1)\n  confusion_matrix(y_true, y_pred)",
    "accuracy_score": "  y_true <- c(1,0,1,1)\n  y_pred <- c(1,0,0,1)\n  accuracy_score(y_true, y_pred)",
    "precision_score": "  y_true <- c(1,0,1,1)\n  y_pred <- c(1,0,0,1)\n  precision_score(y_true, y_pred)",
    "recall_score": "  y_true <- c(1,0,1,1)\n  y_pred <- c(1,0,0,1)\n  recall_score(y_true, y_pred)",
    "f1_score": "  y_true <- c(1,0,1,1)\n  y_pred <- c(1,0,0,1)\n  f1_score(y_true, y_pred)",
    "plot_confusion_matrix": "  cm <- matrix(c(10, 2, 1, 15), nrow=2)\n  try({ plot_confusion_matrix(cm) })",
    "confusion_stats": "  cm <- matrix(c(10, 2, 1, 15), nrow=2)\n  try({ confusion_stats(cm) })",
    "find_best_k": "  x <- matrix(rnorm(200), nrow=100)\n  y <- sample(0:1, 100, replace=TRUE)\n  find_best_k(x, y, k_values=c(1,3,5))",
    "StandardScaler": "  s <- StandardScaler$new()\n  x <- matrix(rnorm(20), nrow=10)\n  s$fit(x)\n  s$transform(x)",
    "MinMaxScaler": "  s <- MinMaxScaler$new()\n  x <- matrix(rnorm(20), nrow=10)\n  s$fit(x)\n  s$transform(x)",
    "RobustScaler": "  s <- RobustScaler$new()\n  x <- matrix(rnorm(20), nrow=10)\n  s$fit(x)\n  s$transform(x)",
    "train_test_split": "  X <- matrix(rnorm(20), nrow=10)\n  y <- sample(0:1, 10, replace=TRUE)\n  train_test_split(X, y, test_size=0.2)",
    "drop_constant_columns": "  x <- data.frame(a=c(1,1,1), b=c(1,2,3))\n  drop_constant_columns(x)",
    "handle_missing_values": "  x <- data.frame(a=c(1,NA,3), b=c(1,2,3))\n  handle_missing_values(x)",
    "fit_linear_model": "  X <- matrix(rnorm(20), nrow=10)\n  y <- rnorm(10)\n  try({ fit_linear_model(X, y) })",
    "predict_linear_model": "  X <- matrix(rnorm(20), nrow=10)\n  y <- rnorm(10)\n  model <- fit_linear_model(X, y)\n  predict_linear_model(model, X)",
    "fit_kmeans": "  X <- matrix(rnorm(20), nrow=10)\n  try({ fit_kmeans(X, k=2) })",
    "predict_kmeans": "  X <- matrix(rnorm(20), nrow=10)\n  model <- fit_kmeans(X, k=2)\n  predict_kmeans(model, X)"
}

for filename in os.listdir(R_DIR):
    if not filename.endswith(".R") or filename == "RcppExports.R":
        continue
    filepath = os.path.join(R_DIR, filename)
    with open(filepath, "r") as f:
        content = f.read()

    blocks = []
    current_block = []
    lines = content.split('\n')
    
    out_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("#'"):
            # collect roxygen block
            block_lines = [line]
            i += 1
            while i < len(lines) and lines[i].startswith("#'"):
                block_lines.append(lines[i])
                i += 1
            
            # Now we have the next line which should be the function/class
            func_line = lines[i] if i < len(lines) else ""
            
            # Check if it has @export
            is_exported = any("@export" in bl for bl in block_lines)
            
            # Extract function/class name
            name = None
            m1 = re.search(r'^([A-Za-z0-9_]+)\s*<-\s*function', func_line)
            m2 = re.search(r'^([A-Za-z0-9_]+)\s*<-\s*setRefClass', func_line)
            if m1:
                name = m1.group(1)
            elif m2:
                name = m2.group(1)
                
            if is_exported and name:
                # modify the block
                has_details = any("@details" in bl for bl in block_lines)
                has_seealso = any("@seealso" in bl for bl in block_lines)
                has_examples = any("@examples" in bl for bl in block_lines)
                has_return = any("@return" in bl for bl in block_lines)
                
                # We want to insert just before @export
                export_idx = -1
                for j, bl in enumerate(block_lines):
                    if "@export" in bl:
                        export_idx = j
                        break
                
                if export_idx == -1:
                    export_idx = len(block_lines) # append at end
                    
                inserts = []
                if not has_details:
                    inserts.append(f"#' @details")
                    inserts.append(f"#' Provides functionality for {escape_latex(name)} operations.")
                if not has_return:
                    inserts.append(f"#' @return A {escape_latex(name)} object or appropriate value")
                if not has_seealso:
                    inserts.append(f"#' @seealso \\code{{\\link{{VectorForgeML-package}}}}")
                if not has_examples and name in EXAMPLES:
                    inserts.append(f"#' @examples")
                    ex_lines = EXAMPLES[name].split('\n')
                    for ex in ex_lines:
                        inserts.append(f"#' {ex}")
                        
                # Ensure spacing
                if inserts:
                    inserts.insert(0, "#'")
                    inserts.append("#'")
                    
                block_lines = block_lines[:export_idx] + inserts + block_lines[export_idx:]
            
            out_lines.extend(block_lines)
        else:
            out_lines.append(line)
            i += 1

    with open(filepath, "w") as f:
        f.write('\n'.join(out_lines))
        
print("Roxygen comments updated.")
