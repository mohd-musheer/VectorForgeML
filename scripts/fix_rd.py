import os
import re
import glob

rd_dir = "e:/VectorForgeML/man"
rd_files = glob.glob(os.path.join(rd_dir, "*.Rd"))

for rd in rd_files:
    with open(rd, "r", encoding="utf-8") as f:
        content = f.read()

    has_details = r"\details{" in content
    has_examples = r"\examples{" in content
    has_value = r"\value{" in content
    has_seealso = r"\seealso{" in content

    func_name = os.path.basename(rd).replace("-class.Rd", "").replace(".Rd", "")

    new_content = "\n"

    if not has_details:
        new_content += "\n\\details{\n  Provides functionality for " + func_name + " operations.\n}\n"
    
    if not has_value:
        new_content += "\n\\value{\n  Returns computed value or object.\n}\n"

    if not has_seealso:
        if func_name != "VectorForgeML-package":
            new_content += "\n\\seealso{\n  \\code{\\link{VectorForgeML-package}}\n}\n"

    if not has_examples:
        ex = "\n\\examples{\n"
        if func_name == "VectorForgeML-package":
            ex += "  library(VectorForgeML)\n"
        elif "Regression" in func_name:
            ex += "  x <- matrix(rnorm(20), nrow=10)\n  y <- rnorm(10)\n  model <- " + func_name + "$new()\n  model$fit(x,y)\n"
        elif any(k in func_name for k in ["Classifier", "Forest", "Tree", "KMeans", "Logistic", "Softmax", "KNN"]):
            ex += "  x <- matrix(rnorm(20), nrow=10)\n  y <- sample(0:1, 10, replace=TRUE)\n  model <- " + func_name + "$new()\n  model$fit(x,y)\n"
        elif any(k in func_name for k in ["Scaler", "Encoder", "PCA"]):
            ex += "  x <- matrix(rnorm(20), nrow=10)\n  model <- " + func_name + "$new()\n  model$fit(x)\n"
        elif any(k in func_name for k in ["Pipeline", "Transformer"]):
            ex += "  model <- " + func_name + "$new()\n"
        elif "split" in func_name:
            ex += "  x <- matrix(rnorm(20), nrow=10)\n  y <- rnorm(10)\n  res <- " + func_name + "(x,y)\n"
        elif "find_best" in func_name:
            ex += "  x <- matrix(rnorm(20), nrow=10)\n  y <- sample(0:1, 10, replace=TRUE)\n  res <- " + func_name + "(x, y)\n"
        else:
            ex += "  y_true <- c(1,0,1,1)\n  y_pred <- c(1,0,0,1)\n  res <- " + func_name + "(y_true, y_pred)\n"
        ex += "}\n"
        new_content += ex

    if new_content.strip():
        with open(rd, "a", encoding="utf-8") as f:
            f.write(new_content)

print("Finished updating Rd files via Python.")
