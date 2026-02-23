#' Fit Linear Model (Fast C++ backend)
#'
#' Internal helper for linear regression training.
#'
#' @param X numeric matrix
#' @param y numeric vector
#'
#' @return model object
#'
#' @details
#' Provides functionality for fit_linear_model operations.
#' @seealso \code{\link{VectorForgeML-package}}
#' @examples
#'   X <- matrix(rnorm(20), nrow=10)
#'   y <- rnorm(10)
#'   try({ fit_linear_model(X, y) })
#'
#' @export
fit_linear_model <- function(X, y) {
  model <- LinearRegression$new()
  model$fit(X, y)
  model
}
#' Predict Linear Model
#'
#' Predict values using trained linear model.
#'
#' @param model trained model
#' @param X matrix
#'
#' @return numeric vector
#'
#' @details
#' Provides functionality for predict_linear_model operations.
#' @seealso \code{\link{VectorForgeML-package}}
#' @examples
#'   X <- matrix(rnorm(20), nrow=10)
#'   y <- rnorm(10)
#'   model <- fit_linear_model(X, y)
#'   predict_linear_model(model, X)
#'
#' @export
predict_linear_model <- function(model, X) {
  if (!methods::is(model, "LinearRegression")) {
    stop("model must be a LinearRegression object")
  }
  model$predict(X)
}
