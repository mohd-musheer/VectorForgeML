import os, glob

for f in glob.glob('e:/VectorForgeML/R/*.R'):
    with open(f, 'r') as file:
        content = file.read()
    
    new_content = content.replace('\\_', '_')
    
    if new_content != content:
        with open(f, 'w') as file:
            file.write(new_content)
        print(f"Removed \\_ from {f}")

