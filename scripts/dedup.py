import os
import glob

rd_dir = "e:/VectorForgeML/man"
for rd in glob.glob(os.path.join(rd_dir, "*.Rd")):
    with open(rd, "r", encoding="utf-8") as f:
        content = f.read()

    # Find the positions of tags
    tags = ["\details{", "\value{", "\seealso{", "\examples{"]
    
    for tag in tags:
        parts = content.split(tag)
        if len(parts) > 2:
            # tag appears more than once
            # Keep the first part, the tag, and the first body.
            # Easiest way to deduplicate identical sections
            # We assume the duplicated sections are exactly the same
            # Let's just find and replace the duplicate block
            
            # Simple dedup:
            pass

    # Actually, a simpler way is regex finding duplicated blocks
    import re
    
    # We can just manually clean the two files we know are messed up.
    # The duplicate in Pipeline
    content = re.sub(r'(\\details\{\s*[^\}]+\s*\})\s*\1', r'\1', content)
    content = re.sub(r'(\\seealso\{\s*[^\}]+\s*\})\s*\1', r'\1', content)
    content = re.sub(r'(\\value\{\s*[^\}]+\s*\})\s*\1', r'\1', content)
    
    with open(rd, "w", encoding="utf-8") as f:
        f.write(content)
print("Dedup done.")
