import os
import re

rd_dir = "e:/VectorForgeML/man"

# Fix OneHotEncoder
with open(os.path.join(rd_dir, "OneHotEncoder-class.Rd"), "r", encoding="utf-8") as f:
    content = f.read()
content = content.replace("x <- matrix(rnorm(20), nrow=10)\n  model <- OneHotEncoder$new()\n  model$fit(x)", 
                          "df <- data.frame(a=c('x','y','x'), b=c('1','2','1'))\n  model <- OneHotEncoder$new()\n  model$fit(df)")
with open(os.path.join(rd_dir, "OneHotEncoder-class.Rd"), "w", encoding="utf-8") as f:
    f.write(content)

# Fix LabelEncoder
with open(os.path.join(rd_dir, "LabelEncoder-class.Rd"), "r", encoding="utf-8") as f:
    content = f.read()
content = content.replace("x <- matrix(rnorm(20), nrow=10)", "x <- c('a', 'b', 'a')")
with open(os.path.join(rd_dir, "LabelEncoder-class.Rd"), "w", encoding="utf-8") as f:
    f.write(content)

# PCA, StandardScaler, MinMaxScaler use matrix so they are fine, but let's check
print("Encoder examples fixed.")
