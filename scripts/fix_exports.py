import os, glob, re

for f in glob.glob('e:/VectorForgeML/R/*.R'):
    with open(f, 'r') as file:
        content = file.read()
    
    # Replace '#' @export\nClassName <- setRefClass' with explicit exports
    pattern = r"#' @export\n([A-Za-z0-9_]+)\s*<-\s*setRefClass"
    replacement = r"#' @export \1\n#' @exportClass \1\n\1 <- setRefClass"
    
    new_content = re.sub(pattern, replacement, content)
    
    if new_content != content:
        with open(f, 'w') as file:
            file.write(new_content)
        print(f"Fixed exports in {f}")

