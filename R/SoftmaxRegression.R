#' Softmax Regression Model
#'
#' Multiclass logistic regression.
#'
#' @return SoftmaxRegression object
#'
#' @details
#' Provides functionality for SoftmaxRegression operations.
#' @seealso \code{\link{VectorForgeML-package}}
#' @examples
#'   model <- SoftmaxRegression$new()
#'   X <- matrix(rnorm(20), nrow=10)
#'   y <- sample(0:2, 10, replace=TRUE)
#'   model$fit(X,y)
#'   model$predict(X)
#'
#' @export SoftmaxRegression
#' @exportClass SoftmaxRegression
SoftmaxRegression <- setRefClass(
  "SoftmaxRegression",
  fields=list(ptr="externalptr"),

  methods=list(

    initialize=function(){
      ptr <<- softmax_create()
    },

    fit=function(X,y){
      X <- as.matrix(X)
      y <- as.integer(y)
      softmax_fit(ptr,X,y)
    },

    predict=function(X){
      X <- as.matrix(X)
      softmax_predict(ptr,X)
    },

    predict_proba=function(X){
      X <- as.matrix(X)
      softmax_predict_proba(ptr,X)
    }
  )
)
