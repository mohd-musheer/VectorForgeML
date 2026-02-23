#' Random Forest Model
#'
#' Ensemble of decision trees.
#'
#' @return RandomForest object
#'
#' @details
#' Provides functionality for RandomForest operations.
#' @seealso \code{\link{VectorForgeML-package}}
#' @examples
#'   model <- RandomForest$new(ntrees=5)
#'   X <- matrix(rnorm(20), nrow=10)
#'   y <- sample(0:1, 10, replace=TRUE)
#'   model$fit(X,y)
#'   model$predict(X)
#'
#' @export RandomForest
#' @exportClass RandomForest
RandomForest <- setRefClass(
  "RandomForest",
  fields=list(ptr="externalptr"),
  methods=list(

    initialize=function(ntrees=50,max_depth=6,mtry=NULL,mode="classification"){
      cls <- mode=="classification"
      ptr <<- rf_create(ntrees,max_depth,
                        if(is.null(mtry)) 3 else mtry,
                        cls)
    },

    fit=function(X,y){
      rf_fit(ptr,as.matrix(X),as.numeric(y))
    },

    predict=function(X){
      rf_predict(ptr,as.matrix(X))
    }
  )
)
