#' KMeans Clustering
#'
#' Unsupervised clustering algorithm.
#'
#' @return KMeans object
#'
#' @details
#' Provides functionality for KMeans operations.
#' @seealso \code{\link{VectorForgeML-package}}
#' @examples
#'   x <- matrix(rnorm(20), nrow=10)
#'   model <- KMeans$new()
#'   model$fit(x)
#'
#' @export KMeans
#' @exportClass KMeans
KMeans <- setRefClass(
  "KMeans",
  fields=list(ptr="externalptr"),
  methods=list(

    initialize=function(k=3){
      ptr <<- kmeans_create(k)
    },

    fit=function(X){
      kmeans_fit(ptr,as.matrix(X))
    },

    predict=function(X){
      kmeans_predict(ptr,as.matrix(X))
    },

    fit_predict=function(X){
      fit(X)
      predict(X)
    }
  )
)
