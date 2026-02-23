#' Ridge Regression Model
#'
#' Linear regression with L2 regularization.
#'
#' @return RidgeRegression object
#'
#' @details
#' Provides functionality for RidgeRegression operations.
#' @seealso \code{\link{VectorForgeML-package}}
#' @examples
#'   model <- RidgeRegression$new()
#'   X <- matrix(rnorm(20), nrow=10)
#'   y <- rnorm(10)
#'   model$fit(X,y,lambda=1.0)
#'   model$predict(X)
#'
#' @export RidgeRegression
#' @exportClass RidgeRegression
RidgeRegression <- setRefClass(
  "RidgeRegression",
  fields=list(ptr="externalptr"),

  methods=list(

    initialize=function(){
      ptr <<- ridge_create()
    },

    fit=function(X,y,lambda=1.0){
      X <- as.matrix(X)
      y <- as.numeric(y)
      ridge_fit(ptr,X,y,lambda)
    },

    predict=function(X){
      X <- as.matrix(X)
      ridge_predict(ptr,X)
    }
  )
)
