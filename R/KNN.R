#' K-Nearest Neighbors Model
#'
#' Instance-based learning algorithm.
#'
#' @return KNN object
#'
#' @details
#' Provides functionality for KNN operations.
#' @seealso \code{\link{VectorForgeML-package}}
#' @examples
#'   model <- KNN$new(k=3, mode="classification")
#'   X <- matrix(rnorm(20), nrow=10)
#'   y <- sample(0:1, 10, replace=TRUE)
#'   model$fit(X,y)
#'   model$predict(X)
#'
#' @export KNN
#' @exportClass KNN
KNN <- setRefClass(
  "KNN",
  fields=list(ptr="externalptr"),

  methods=list(

    initialize=function(k=5, mode="classification"){

      m <- if(mode=="classification") 0 else 1
      ptr <<- knn_create(k,m)
    },

    fit=function(X,y){
      X <- as.matrix(X)
      y <- as.numeric(y)
      knn_fit(ptr,X,y)
    },

    predict=function(X){
      X <- as.matrix(X)
      knn_predict(ptr,X)
    }
  )
)
