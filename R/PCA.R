#' Principal Component Analysis
#'
#' Dimensionality reduction technique.
#'
#' @return PCA object
#'
#' @details
#' Provides functionality for PCA operations.
#' @seealso \code{\link{VectorForgeML-package}}
#' @examples
#'   model <- PCA$new(n_components=2)
#'   X <- matrix(rnorm(30), nrow=10)
#'   model$fit(X)
#'   model$transform(X)
#'
#' @export PCA
#' @exportClass PCA
PCA <- setRefClass(
  "PCA",
  fields=list(ptr="externalptr", ncomp="numeric"),
  methods=list(
    initialize=function(n_components=2){
      ncomp <<- n_components
      ptr <<- pca_create(n_components)
    },
    fit=function(X){
      pca_fit(ptr,as.matrix(X))
    },
    transform=function(X){
      pca_transform(ptr,as.matrix(X))
    },
    fit_transform=function(X){
      fit(X)
      transform(X)
    }
  )
)
