export class Navbar {
    constructor() {
        this.container = document.getElementById('navbar');
    }

    render() {
        if (!this.container) return;
        this.container.innerHTML = `
            <div class="nav-container">
                <a href="/" class="logo-area">
                    <img src="/assets/images/VectorForgeML_Logo.png" alt="VectorForgeML Logo">
                    <span>VectorForgeML</span>
                </a>
                <button class="menu-toggle" id="menu-toggle">☰</button>
                <div class="nav-links" id="nav-links">
                    <div class="search-box" style="position: relative; margin-right: 1rem;">
                        <input type="text" id="global-search" placeholder="Search docs..." style="
                            background: rgba(255,255,255,0.1);
                            border: 1px solid var(--glass-border);
                            color: white;
                            padding: 0.5rem 1rem;
                            border-radius: 6px;
                            width: 200px;
                            transition: width 0.3s;
                        ">
                        <div id="search-results" style="
                            position: absolute;
                            top: 100%;
                            left: 0;
                            width: 100%;
                            background: var(--card-bg);
                            border: 1px solid var(--glass-border);
                            border-radius: 6px;
                            max-height: 300px;
                            overflow-y: auto;
                            display: none;
                            z-index: 1001;
                        "></div>
                    </div>
                    <a href="/docs/">Docs</a>
                    <a href="/install/">Install</a>
                    <a href="/playground/">Playground</a>
                    <a href="/community/">Community</a>
                    <a href="https://github.com/mohd-musheer/VectorForgeML" target="_blank" class="btn-primary" style="padding: 0.5rem 1rem; margin-left: 1rem;">GitHub</a>
                </div>
            </div>
        `;
        this.attachEvents();
        // Dynamic import to avoid module issues if search.js isn't loaded yet, or just rely on main.js to handle logic
        // For simplicity, we'll dispatch an event meant for search.js
    }

    attachEvents() {
        const toggle = document.getElementById('menu-toggle');
        const links = document.getElementById('nav-links');
        const searchInput = document.getElementById('global-search');

        if (toggle && links) {
            toggle.addEventListener('click', () => {
                links.classList.toggle('mobile-open');
            });
        }

        if (searchInput) {
            searchInput.addEventListener('input', (e) => {
                document.dispatchEvent(new CustomEvent('search-input', { detail: e.target.value }));
            });
            searchInput.addEventListener('focus', () => {
                document.dispatchEvent(new CustomEvent('search-focus'));
            });
        }
    }
}
