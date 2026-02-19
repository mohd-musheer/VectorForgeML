const fs = require('fs');
const path = require('path');

const DOCS_DIR = path.join(__dirname, '..', 'docs', 'docs');

// Get all HTML files in docs/docs
const files = fs.readdirSync(DOCS_DIR).filter(f => f.endsWith('.html') && f !== 'index.html');

files.forEach(file => {
    const filePath = path.join(DOCS_DIR, file);
    let content = fs.readFileSync(filePath, 'utf8');

    // Check if sections exist
    if (content.includes('<h3>Parameters</h3>')) {
        console.log(`Skipping ${file} - already structured`);
        return;
    }

    // We want to insert these sections.
    // Optimal location: After Description key facts (usually the first glass-panel/section).
    // Or at the end?
    // User order: Title, Description, Parameters, Usage, Output, Complexity, How it works, Notes.
    // Current: Title, Description, [Workflow], [Impl], [Complexity], [UseCases].

    // Strategy: Insert Parameters, Usage, Output AFTER Description.
    // Insert "How it works" if "Algorithm Workflow" is missing, or rename it?
    // User said "How it works section (leave placeholder)".
    // I will add unique placeholders.

    const injection = `
            <!-- Injected Structure -->
            <section class="slide-up" style="animation-delay: 0.1s; margin-top: 3rem;">
                <h3>Parameters</h3>
                <div class="glass-panel" style="padding: 1rem;">
                    <table style="margin:0; border:none;">
                        <tr><th style="padding-left:0;">Name</th><th>Type</th><th>Description</th></tr>
                        <tr><td style="padding-left:0;"><code>x</code></td><td>Matrix</td><td>Input feature matrix.</td></tr>
                        <tr><td style="padding-left:0;"><code>y</code></td><td>Vector</td><td>Target vector.</td></tr>
                        <tr><td style="padding-left:0; border-bottom:none;"><code>...</code></td><td style="border-bottom:none;">...</td><td style="border-bottom:none;">Additional parameters.</td></tr>
                    </table>
                </div>
            </section>

            <section class="slide-up" style="animation-delay: 0.15s; margin-top: 3rem;">
                <h3>Usage Example</h3>
                <div class="glass-panel" style="padding: 0; overflow: hidden;">
                    <pre><code class="language-r"># Load library
library(VectorForgeML)

# Prepare data
X <- matrix(rnorm(100), ncol=5)
y <- rnorm(20)

# Initialize and train
model <- ${file.replace('.html', '')}()
model$fit(X, y)

# Predict
preds <- model$predict(X)</code></pre>
                </div>
            </section>

            <section class="slide-up" style="animation-delay: 0.2s; margin-top: 3rem;">
                <h3>Output Example</h3>
                <div class="glass-panel" style="padding: 1.5rem; font-family: 'Fira Code', monospace; color: var(--text-secondary); font-size: 0.9rem;">
                    [1] 0.234 0.567 -0.123 ...
                </div>
            </section>
            
            <section class="slide-up" style="animation-delay: 0.25s; margin-top: 3rem;">
                <h3>How it Works</h3>
                 <div class="glass-panel" style="padding: 2rem;">
                    <p style="color: var(--text-secondary);">The internal mechanism involves optimized C++ routines...</p>
                </div>
            </section>
    `;

    const notes = `
            <section class="slide-up" style="animation-delay: 0.5s; margin-top: 3rem;">
                <h3>Notes</h3>
                <div class="glass-panel" style="padding: 1.5rem; border-left: 4px solid var(--accent-secondary);">
                    <p style="color: var(--text-secondary); margin:0;">Performance scales linearly with input size.</p>
                </div>
            </section>
    `;

    // Insert Injection AFTER the first <section> (Description)
    // Find first </section>
    const firstSectionEnd = content.indexOf('</section>');
    if (firstSectionEnd !== -1) {
        content = content.slice(0, firstSectionEnd + 10) + injection + content.slice(firstSectionEnd + 10);
    }

    // Insert Notes at the end of <main>, before </main>
    const mainEnd = content.indexOf('</main>');
    if (mainEnd !== -1) {
        content = content.slice(0, mainEnd) + notes + content.slice(mainEnd);
    }

    fs.writeFileSync(filePath, content);
    console.log(`Structured: ${file}`);
});
