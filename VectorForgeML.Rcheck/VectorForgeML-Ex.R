pkgname <- "VectorForgeML"
source(file.path(R.home("share"), "R", "examples-header.R"))
options(warn = 1)
options(pager = "console")
base::assign(".ExTimings", "VectorForgeML-Ex.timings", pos = 'CheckExEnv')
base::cat("name\tuser\tsystem\telapsed\n", file=base::get(".ExTimings", pos = 'CheckExEnv'))
base::assign(".format_ptime",
function(x) {
  if(!is.na(x[4L])) x[1L] <- x[1L] + x[4L]
  if(!is.na(x[5L])) x[2L] <- x[2L] + x[5L]
  options(OutDec = '.')
  format(x[1L:3L], digits = 7L)
},
pos = 'CheckExEnv')

### * </HEADER>
library('VectorForgeML')

base::assign(".oldSearch", base::search(), pos = 'CheckExEnv')
base::assign(".old_wd", base::getwd(), pos = 'CheckExEnv')
cleanEx()
nameEx("ColumnTransformer-class")
### * ColumnTransformer-class

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: ColumnTransformer-class
### Title: Column Transformer
### Aliases: ColumnTransformer-class ColumnTransformer

### ** Examples

  model <- ColumnTransformer$new(num_cols="A", cat_cols="B")




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("ColumnTransformer-class", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("DecisionTree-class")
### * DecisionTree-class

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: DecisionTree-class
### Title: Decision Tree Model
### Aliases: DecisionTree-class DecisionTree

### ** Examples

  model <- DecisionTree$new()
  X <- matrix(rnorm(20), nrow=10)
  y <- sample(0:1, 10, replace=TRUE)
  model$fit(X,y)
  model$predict(X)




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("DecisionTree-class", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("KMeans-class")
### * KMeans-class

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: KMeans-class
### Title: KMeans Clustering
### Aliases: KMeans-class KMeans

### ** Examples

  x <- matrix(rnorm(20), nrow=10)
  model <- KMeans$new()
  model$fit(x)




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("KMeans-class", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("KNN-class")
### * KNN-class

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: KNN-class
### Title: K-Nearest Neighbors Model
### Aliases: KNN-class KNN

### ** Examples

  model <- KNN$new(k=3, mode="classification")
  X <- matrix(rnorm(20), nrow=10)
  y <- sample(0:1, 10, replace=TRUE)
  model$fit(X,y)
  model$predict(X)




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("KNN-class", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("LabelEncoder-class")
### * LabelEncoder-class

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: LabelEncoder-class
### Title: Label Encoder
### Aliases: LabelEncoder-class LabelEncoder

### ** Examples

  enc <- LabelEncoder$new()
  x <- c("a", "b", "a")
  enc$fit(x)
  enc$transform(x)




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("LabelEncoder-class", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("LinearRegression-class")
### * LinearRegression-class

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: LinearRegression-class
### Title: Linear Regression Model
### Aliases: LinearRegression-class LinearRegression

### ** Examples

model <- LinearRegression$new()
X <- matrix(rnorm(100),50,2)
y <- rnorm(50)
model$fit(X,y)
model$predict(X)





base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("LinearRegression-class", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("LogisticRegression-class")
### * LogisticRegression-class

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: LogisticRegression-class
### Title: Logistic Regression Model
### Aliases: LogisticRegression-class LogisticRegression

### ** Examples

  model <- LogisticRegression$new()
  X <- matrix(rnorm(20), nrow=10)
  y <- sample(0:1, 10, replace=TRUE)
  model$fit(X,y)
  model$predict(X)




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("LogisticRegression-class", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("MinMaxScaler-class")
### * MinMaxScaler-class

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: MinMaxScaler-class
### Title: Standard Scaler
### Aliases: MinMaxScaler-class MinMaxScaler

### ** Examples

  s <- MinMaxScaler$new()
  x <- matrix(rnorm(20), nrow=10)
  s$fit(x)
  s$transform(x)




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("MinMaxScaler-class", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("OneHotEncoder-class")
### * OneHotEncoder-class

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: OneHotEncoder-class
### Title: One Hot Encoder
### Aliases: OneHotEncoder-class OneHotEncoder

### ** Examples

  enc <- OneHotEncoder$new()
  df <- data.frame(a=c("x","y","x"))
  enc$fit(df)
  enc$transform(df)




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("OneHotEncoder-class", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("PCA-class")
### * PCA-class

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: PCA-class
### Title: Principal Component Analysis
### Aliases: PCA-class PCA

### ** Examples

  model <- PCA$new(n_components=2)
  X <- matrix(rnorm(30), nrow=10)
  model$fit(X)
  model$transform(X)




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("PCA-class", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("Pipeline-class")
### * Pipeline-class

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: Pipeline-class
### Title: Pipeline
### Aliases: Pipeline-class Pipeline

### ** Examples

  model <- Pipeline$new(list(StandardScaler$new()))




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("Pipeline-class", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("RandomForest-class")
### * RandomForest-class

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: RandomForest-class
### Title: Random Forest Model
### Aliases: RandomForest-class RandomForest

### ** Examples

  model <- RandomForest$new(ntrees=5)
  X <- matrix(rnorm(20), nrow=10)
  y <- sample(0:1, 10, replace=TRUE)
  model$fit(X,y)
  model$predict(X)




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("RandomForest-class", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("RidgeRegression-class")
### * RidgeRegression-class

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: RidgeRegression-class
### Title: Ridge Regression Model
### Aliases: RidgeRegression-class RidgeRegression

### ** Examples

  model <- RidgeRegression$new()
  X <- matrix(rnorm(20), nrow=10)
  y <- rnorm(10)
  model$fit(X,y,lambda=1.0)
  model$predict(X)




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("RidgeRegression-class", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("SoftmaxRegression-class")
### * SoftmaxRegression-class

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: SoftmaxRegression-class
### Title: Softmax Regression Model
### Aliases: SoftmaxRegression-class SoftmaxRegression

### ** Examples

  model <- SoftmaxRegression$new()
  X <- matrix(rnorm(20), nrow=10)
  y <- sample(0:2, 10, replace=TRUE)
  model$fit(X,y)
  model$predict(X)




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("SoftmaxRegression-class", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("StandardScaler-class")
### * StandardScaler-class

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: StandardScaler-class
### Title: Drop Constant Columns
### Aliases: StandardScaler-class StandardScaler

### ** Examples

  s <- StandardScaler$new()
  x <- matrix(rnorm(20), nrow=10)
  s$fit(x)
  s$transform(x)




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("StandardScaler-class", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("accuracy_score")
### * accuracy_score

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: accuracy_score
### Title: Accuracy Score
### Aliases: accuracy_score

### ** Examples

  y_true <- c(1,0,1,1)
  y_pred <- c(1,0,0,1)
  accuracy_score(y_true, y_pred)




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("accuracy_score", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("confusion_matrix")
### * confusion_matrix

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: confusion_matrix
### Title: Confusion Matrix
### Aliases: confusion_matrix

### ** Examples

  y_true <- c(1,0,1,1)
  y_pred <- c(1,0,0,1)
  confusion_matrix(y_true, y_pred)




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("confusion_matrix", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("confusion_stats")
### * confusion_stats

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: confusion_stats
### Title: Confusion Matrix Statistics
### Aliases: confusion_stats

### ** Examples

  cm <- matrix(c(10, 2, 1, 15), nrow=2)
  try({ confusion_stats(cm) })




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("confusion_stats", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("drop_constant_columns")
### * drop_constant_columns

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: drop_constant_columns
### Title: Drop Constant Columns
### Aliases: drop_constant_columns

### ** Examples

  x <- data.frame(a=c(1,1,1), b=c(1,2,3))
  drop_constant_columns(x)




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("drop_constant_columns", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("f1_score")
### * f1_score

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: f1_score
### Title: F1 Score
### Aliases: f1_score

### ** Examples

  y_true <- c(1,0,1,1)
  y_pred <- c(1,0,0,1)
  f1_score(y_true, y_pred)




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("f1_score", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("find_best_k")
### * find_best_k

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: find_best_k
### Title: Find Best K
### Aliases: find_best_k

### ** Examples

  x <- matrix(rnorm(200), nrow=100)
  y <- sample(0:1, 100, replace=TRUE)
  find_best_k(x, y, k_values=c(1,3,5))




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("find_best_k", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("fit_linear_model")
### * fit_linear_model

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: fit_linear_model
### Title: Fit Linear Model (Fast C++ backend)
### Aliases: fit_linear_model

### ** Examples

  X <- matrix(rnorm(20), nrow=10)
  y <- rnorm(10)
  try({ fit_linear_model(X, y) })




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("fit_linear_model", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("plot_confusion_matrix")
### * plot_confusion_matrix

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: plot_confusion_matrix
### Title: Plot Confusion Matrix
### Aliases: plot_confusion_matrix

### ** Examples

  cm <- matrix(c(10, 2, 1, 15), nrow=2)
  try({ plot_confusion_matrix(cm) })




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("plot_confusion_matrix", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("precision_score")
### * precision_score

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: precision_score
### Title: Precision Score
### Aliases: precision_score

### ** Examples

  y_true <- c(1,0,1,1)
  y_pred <- c(1,0,0,1)
  precision_score(y_true, y_pred)




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("precision_score", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("predict_linear_model")
### * predict_linear_model

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: predict_linear_model
### Title: Predict Linear Model
### Aliases: predict_linear_model

### ** Examples

  X <- matrix(rnorm(20), nrow=10)
  y <- rnorm(10)
  model <- fit_linear_model(X, y)
  predict_linear_model(model, X)




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("predict_linear_model", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("recall_score")
### * recall_score

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: recall_score
### Title: Recall Score
### Aliases: recall_score

### ** Examples

  y_true <- c(1,0,1,1)
  y_pred <- c(1,0,0,1)
  recall_score(y_true, y_pred)




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("recall_score", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("train_test_split")
### * train_test_split

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: train_test_split
### Title: Train Test Split
### Aliases: train_test_split

### ** Examples

  X <- matrix(rnorm(20), nrow=10)
  y <- sample(0:1, 10, replace=TRUE)
  train_test_split(X, y, test_size=0.2)




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("train_test_split", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
### * <FOOTER>
###
cleanEx()
options(digits = 7L)
base::cat("Time elapsed: ", proc.time() - base::get("ptime", pos = 'CheckExEnv'),"\n")
grDevices::dev.off()
###
### Local variables: ***
### mode: outline-minor ***
### outline-regexp: "\\(> \\)?### [*]+" ***
### End: ***
quit('no')
