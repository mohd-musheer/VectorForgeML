library(VectorForgeML)

df <- read.csv("inst/dataset/winequality.csv",sep=";")

y <- df$quality
X <- df; X$quality<-NULL

# Split
split <- train_test_split(X, y, 0.2, 42)

# Column detection
cat_cols <- names(X)[sapply(X, is.character)]
num_cols <- names(X)[!sapply(X, is.character)]

# Pipeline Setup
pre <- ColumnTransformer$new(
  num_cols = num_cols,
  cat_cols = cat_cols,
  num_pipeline = StandardScaler$new(),
  cat_pipeline = OneHotEncoder$new()
)

# RandomForest specific parameters: ntrees=100, max_depth=7
pipe <- Pipeline$new(list(
  pre,
  RandomForest$new(ntrees=100, max_depth=7, mtry=4, mode="classification")
))

# Benchmark Training Time
start_time <- Sys.time()
pipe$fit(split$X_train, split$y_train)
end_time <- Sys.time()

# Predict and Evaluate
pred <- pipe$predict(split$X_test)

cat("\n--- VectorForgeML Performance ---\n")
cat("Train Time:", as.numeric(end_time - start_time), "seconds\n")
cat("Accuracy:  ", accuracy_score(split$y_test, round(pred)), "\n")