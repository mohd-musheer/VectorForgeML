export class Footer {
    constructor() {
        this.container = document.getElementById('footer');
    }

    render() {
        if (!this.container) return;
        this.container.innerHTML = `
            <footer>
                <div class="footer-grid">
                    <div class="footer-col">
                        <h4>VectorForgeML</h4>
                        <p style="color: var(--text-secondary);">High-performance machine learning for R & C++.</p>
                        <p style="margin-top: 1rem;">© ${new Date().getFullYear()} VectorForgeML</p>
                    </div>
                    <div class="footer-col">
                        <h4>Resources</h4>
                        <ul>
                            <li><a href="/docs/">Documentation</a></li>
                            <li><a href="/install/">Installation</a></li>
                            <li><a href="/playground/">Playground</a></li>
                        </ul>
                    </div>
                    <div class="footer-col">
                        <h4>Community</h4>
                        <ul>
                            <li><a href="https://github.com/mohd-musheer/VectorForgeML">GitHub</a></li>
                            <li><a href="/community/">Discord</a></li>
                            <li><a href="/developer/">Contributing</a></li>
                        </ul>
                    </div>
                    <div class="footer-col">
                        <h4>Legal</h4>
                        <ul>
                            <li><a href="/license.html">License</a></li>
                            <li><a href="#">Privacy Policy</a></li>
                        </ul>
                    </div>
                </div>
            </footer>
        `;
    }
}
