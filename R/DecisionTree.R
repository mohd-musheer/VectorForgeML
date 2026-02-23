#' Decision Tree Model
#'
#' Tree-based classification/regression algorithm.
#'
#' @return DecisionTree object
#'
#' @details
#' Provides functionality for DecisionTree operations.
#' @seealso \code{\link{VectorForgeML-package}}
#' @examples
#'   model <- DecisionTree$new()
#'   X <- matrix(rnorm(20), nrow=10)
#'   y <- sample(0:1, 10, replace=TRUE)
#'   model$fit(X,y)
#'   model$predict(X)
#'
#' @export DecisionTree
#' @exportClass DecisionTree
DecisionTree <- setRefClass(
  "DecisionTree",
  fields=list(ptr="externalptr"),
  methods=list(
    initialize=function(max_depth=5){
      ptr <<- dt_create(max_depth)
    },
    fit=function(X,y){
      dt_fit(ptr,as.matrix(X),as.numeric(y))
    },
    predict=function(X){
      dt_predict(ptr,as.matrix(X))
    }
  )
)
