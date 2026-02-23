#' Logistic Regression Model
#'
#' Binary classification logistic regression.
#'
#' @return LogisticRegression object
#'
#' @details
#' Provides functionality for LogisticRegression operations.
#' @seealso \code{\link{VectorForgeML-package}}
#' @examples
#'   model <- LogisticRegression$new()
#'   X <- matrix(rnorm(20), nrow=10)
#'   y <- sample(0:1, 10, replace=TRUE)
#'   model$fit(X,y)
#'   model$predict(X)
#'
#' @export LogisticRegression
#' @exportClass LogisticRegression
LogisticRegression <- setRefClass(
  "LogisticRegression",
  fields=list(ptr="externalptr"),

  methods=list(

    initialize=function(){
      ptr <<- logreg_create()
    },

    fit=function(X,y){
      X <- as.matrix(X)
      y <- as.numeric(y)
      logreg_fit(ptr,X,y)
    },

    predict=function(X){
      X <- as.matrix(X)
      logreg_predict(ptr,X)
    },

    predict_proba=function(X){
      X <- as.matrix(X)
      logreg_predict_proba(ptr,X)
    }
  )
)
