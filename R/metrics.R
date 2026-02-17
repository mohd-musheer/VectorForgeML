mse <- function(y_true,y_pred){
  mean((y_true-y_pred)^2, na.rm=TRUE)
}

rmse <- function(y_true,y_pred){
  sqrt(mse(y_true,y_pred))
}


r2_score <- function(y_true,y_pred){

  y_true <- y_true[!is.na(y_pred)]
  y_pred <- y_pred[!is.na(y_pred)]

  ss_res <- sum((y_true-y_pred)^2)
  ss_tot <- sum((y_true-mean(y_true))^2)

  if(is.na(ss_tot) || ss_tot==0)
    return(1)

  1 - ss_res/ss_tot
}

# =========================
# CLASSIFICATION METRICS
# =========================

accuracy_score <- function(y_true, y_pred){
  y_true <- as.vector(y_true)
  y_pred <- as.vector(y_pred)
  mean(y_true == y_pred, na.rm = TRUE)
}


precision_score <- function(y_true, y_pred, positive = NULL){
  y_true <- as.vector(y_true)
  y_pred <- as.vector(y_pred)

  if(is.null(positive))
    positive <- unique(y_true)[1]

  tp <- sum(y_true == positive & y_pred == positive)
  fp <- sum(y_true != positive & y_pred == positive)

  if(tp + fp == 0) return(0)
  tp/(tp+fp)
}


recall_score <- function(y_true, y_pred, positive = NULL){
  y_true <- as.vector(y_true)
  y_pred <- as.vector(y_pred)

  if(is.null(positive))
    positive <- unique(y_true)[1]

  tp <- sum(y_true == positive & y_pred == positive)
  fn <- sum(y_true == positive & y_pred != positive)

  if(tp + fn == 0) return(0)
  tp/(tp+fn)
}


f1_score <- function(y_true, y_pred, positive = NULL){
  p <- precision_score(y_true, y_pred, positive)
  r <- recall_score(y_true, y_pred, positive)

  if(p+r == 0) return(0)
  2*p*r/(p+r)
}


# =========================
# MULTICLASS MACRO METRICS
# =========================

macro_precision <- function(y_true, y_pred){
  classes <- unique(y_true)
  mean(sapply(classes, function(cls)
    precision_score(y_true, y_pred, cls)))
}

macro_recall <- function(y_true, y_pred){
  classes <- unique(y_true)
  mean(sapply(classes, function(cls)
    recall_score(y_true, y_pred, cls)))
}

macro_f1 <- function(y_true, y_pred){
  classes <- unique(y_true)
  mean(sapply(classes, function(cls)
    f1_score(y_true, y_pred, cls)))
}

confusion_matrix <- function(y_true, y_pred){

  y_true <- as.vector(y_true)
  y_pred <- as.vector(y_pred)

  classes <- sort(unique(c(y_true,y_pred)))
  k <- length(classes)

  mat <- matrix(0L,k,k,
                dimnames=list(
                  Actual=classes,
                  Predicted=classes
                ))

  for(i in seq_along(y_true)){
    a <- match(y_true[i],classes)
    p <- match(y_pred[i],classes)
    mat[a,p] <- mat[a,p] + 1L
  }

  mat
}


confusion_stats <- function(cm){

  total <- sum(cm)
  acc <- sum(diag(cm))/total

  precision <- diag(cm)/colSums(cm)
  recall <- diag(cm)/rowSums(cm)
  f1 <- 2*precision*recall/(precision+recall)

  list(
    accuracy = acc,
    precision = precision,
    recall = recall,
    f1 = f1
  )
}

plot_confusion_matrix <- function(cm, normalize=TRUE){

  if(normalize)
    cm <- cm / rowSums(cm)

  classes <- colnames(cm)
  n <- nrow(cm)

  palette <- colorRampPalette(c("#3B4CC0","#FFFFFF","#B40426"))(100)

  par(mar=c(5,5,2,2))

  image(
    1:n,
    1:n,
    t(cm[n:1,]),
    col=palette,
    axes=FALSE,
    xlab="Predicted",
    ylab="Actual"
  )

  axis(1,1:n,classes)
  axis(2,1:n,rev(classes))

  # grid lines
  abline(h=seq(0.5,n+0.5,1), col="gray85")
  abline(v=seq(0.5,n+0.5,1), col="gray85")

  # annotations
  for(i in 1:n){
    for(j in 1:n){
      text(j, n-i+1,
           sprintf("%.2f",cm[i,j]),
           font=2,
           cex=0.9)
    }
  }
}
