
#' Drop Constant Columns
#'
#' Removes columns with zero variance.
#'
#' @param X input matrix/dataframe
#' @param eps for param eps
#' @return cleaned matrix
#'
#' @details
#' Provides functionality for drop_constant_columns operations.
#' @seealso \code{\link{VectorForgeML-package}}
#' @examples
#'   x <- data.frame(a=c(1,1,1), b=c(1,2,3))
#'   drop_constant_columns(x)
#'
#' @export
drop_constant_columns <- function(X, eps = 1e-12) {
  X <- as.matrix(X)
  cpp_drop_constant_cols(X, eps = eps)
}
