#' Find Best K
#'
#' Finds optimal K value for KNN.
#'
#' @param X features
#' @param y labels
#' @param k_values for k value
#' @return numeric best k
#'
#' @details
#' Provides functionality for find_best_k operations.
#' @seealso \code{\link{VectorForgeML-package}}
#' @examples
#'   x <- matrix(rnorm(200), nrow=100)
#'   y <- sample(0:1, 100, replace=TRUE)
#'   find_best_k(x, y, k_values=c(1,3,5))
#'
#' @export
find_best_k <- function(X,y,k_values=seq(1,15,2)){

  split <- train_test_split(X,y,0.2,42)

  best_k <- k_values[1]
  best_score <- -Inf

  for(k in k_values){

    model <- KNN$new(k=k,mode="classification")

    model$fit(split$X_train,split$y_train)

    pred <- model$predict(split$X_test)

    score <- accuracy_score(split$y_test,pred)

    if(score > best_score){
      best_score <- score
      best_k <- k
    }
  }

  best_k
}
