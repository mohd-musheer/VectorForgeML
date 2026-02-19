export class Navbar {
    constructor() {
        this.container = document.getElementById('navbar');
    }

    // Helper to calculate relative path relative to root
    getRelativeRoot() {
        const path = window.location.pathname;
        // Count how many folders deep we are
        // / -> 0 (actually 1 empty slot)
        // /docs/LinearRegression.html -> 2
        // /install/index.html -> 2

        // Remove leading slash and split
        const parts = path.split('/').filter(p => p.length > 0);

        // If we are at root index.html or just /
        if (parts.length === 0 || (parts.length === 1 && parts[0].endsWith('.html'))) {
            return './';
        }

        // If we are in a subdir like /docs/algo.html, we need ../
        // Special case for ending with a file
        let depth = parts.length;
        if (parts[parts.length - 1].includes('.html')) {
            // e.g. /docs/algo.html -> we are inside docs/
            // we want to go up 1 level
        }

        // Robust way: assume strict structure:
        // Root is parent of 'assets', 'docs', 'install'.
        // Check if we are in 'docs', 'install', 'playground', 'community', 'developer'

        if (parts.includes('docs') || parts.includes('install') || parts.includes('playground') || parts.includes('community') || parts.includes('developer')) {
            return '../';
        }

        return './';
    }

    render() {
        if (!this.container) return;

        const root = this.getRelativeRoot();

        this.container.innerHTML = `
            <div class="nav-container">
                <a href="${root}index.html" class="logo-area">
                    <img src="${root}assets/images/VectorForgeML_Logo.png" alt="VectorForgeML Logo">
                    <span>VectorForgeML</span>
                </a>
                <button class="menu-toggle" id="menu-toggle" aria-label="Toggle Menu">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="3" y1="12" x2="21" y2="12"></line><line x1="3" y1="6" x2="21" y2="6"></line><line x1="3" y1="18" x2="21" y2="18"></line></svg>
                </button>
                <div class="nav-links" id="nav-links">
                    <a href="${root}docs/index.html">Docs</a>
                    <a href="${root}install/index.html">Install</a>
                    <a href="${root}playground/index.html">Playground</a>
                    <a href="${root}community/index.html">Community</a>
                    <a href="https://github.com/mohd-musheer/VectorForgeML" target="_blank" class="btn-primary" style="padding: 0.5rem 1rem;">GitHub</a>
                </div>
            </div>
        `;
        this.attachEvents();
    }

    attachEvents() {
        const toggle = document.getElementById('menu-toggle');
        const links = document.getElementById('nav-links');

        // Sidebar toggle logic if sidebar exists
        const sidebar = document.getElementById('sidebar');

        if (toggle && links) {
            toggle.addEventListener('click', () => {
                links.classList.toggle('mobile-open');

                // If sidebar exists, we might want to toggle it too on mobile?
                // Usually sidebar has its own toggle or shares this one.
                // Let's try to find the Sidebar instance or manually toggle the class
                if (sidebar) {
                    sidebar.classList.toggle('mobile-open');
                    const overlay = document.getElementById('sidebar-overlay');
                    if (overlay) overlay.style.display = sidebar.classList.contains('mobile-open') ? 'block' : 'none';
                }
            });
        }
    }
}
