export class Sidebar {
    constructor() {
        this.container = document.getElementById('sidebar');
        // Define base path relative to current location
        this.basePath = this.getPathPrefix();
    }

    getPathPrefix() {
        // Simple heuristic: count depth from root
        const path = window.location.pathname;
        if (path.includes('/docs/')) {
            return './'; // We are already in docs
        }
        return './docs/'; // Fallback
    }

    // Helper to get correct link path
    getLink(algo) {
        // If we represent the "current" page, we want the link to be usable.
        // Assuming we are always serving flat HTMLs in /docs/ or /public/docs/
        // If we are in /docs/, we can link to "algo.html"
        // If we are in root, we link to "docs/algo.html"

        // Robust relative path strategy:
        // 1. Get current directory path array
        const parts = window.location.pathname.split('/').filter(p => p.length > 0);

        // Check if we are in 'docs'
        const inDocs = parts.includes('docs');

        if (inDocs) {
            return `${algo}.html`;
        } else {
            return `docs/${algo}.html`;
        }
    }

    render() {
        if (!this.container) return;

        const path = window.location.pathname;
        const currentFile = path.split('/').pop() || 'index.html';

        const groups = {
            "Models": [
                "DecisionTree", "RandomForest", "LinearRegression", "LogisticRegression",
                "SoftmaxRegression", "RidgeRegression", "KNN", "KMeans", "PCA"
            ],
            "Preprocessing": [
                "StandardScaler", "MinMaxScaler", "drop_constant_columns", "LabelEncoder",
                "OneHotEncoder", "ColumnTransformer", "Pipeline", "train_test_split"
            ],
            "Metrics": [
                "accuracy_score", "precision_score", "recall_score", "f1_score", "macro_f1",
                "macro_precision", "macro_recall", "confusion_matrix", "confusion_stats",
                "mse", "rmse", "r2_score"
            ],
            "Core Engine / Utilities": [
                "dot_product", "square_vec", "cpp_sum_squares", "matrix_ops", "cpp_set_blas_threads"
            ]
        };

        // Determine "Intro" and "Install" links based on where we are
        // If in docs/: index.html, ../install/index.html
        // If in root/: docs/index.html, install/index.html

        const inDocs = path.includes('/docs/');
        const introLink = inDocs ? "index.html" : "docs/index.html";
        const installLink = inDocs ? "../install/index.html" : "install/index.html";

        let sidebarHTML = `
            <div class="sidebar-group">
                <h3>Getting Started</h3>
                <a href="${introLink}" class="sidebar-link ${currentFile === 'index.html' && inDocs ? 'active' : ''}">Introduction</a>
                <a href="${installLink}" class="sidebar-link">Installation</a>
            </div>
        `;

        for (const [groupName, items] of Object.entries(groups)) {
            const links = items.map(algo => {
                const linkHref = inDocs ? `${algo}.html` : `docs/${algo}.html`;
                const isActive = currentFile === `${algo}.html` ? 'active' : '';
                return `<a href="${linkHref}" class="sidebar-link ${isActive}">${algo}</a>`;
            }).join('');

            sidebarHTML += `
                <div class="sidebar-group">
                    <h3>${groupName}</h3>
                    ${links}
                </div>
            `;
        }

        this.container.innerHTML = sidebarHTML;

        this.addMobileOverlay();
    }

    addMobileOverlay() {
        // Add overlay for mobile if not exists
        if (!document.getElementById('sidebar-overlay')) {
            const overlay = document.createElement('div');
            overlay.id = 'sidebar-overlay';
            overlay.style.cssText = `
                position: fixed; top: 0; left: 0; width: 100%; height: 100%;
                background: rgba(0,0,0,0.5); z-index: 1500; display: none;
                backdrop-filter: blur(2px);
            `;
            document.body.appendChild(overlay);

            overlay.addEventListener('click', () => {
                this.container.classList.remove('mobile-open');
                overlay.style.display = 'none';
            });
        }
    }

    toggleMobile() {
        this.container.classList.toggle('mobile-open');
        const overlay = document.getElementById('sidebar-overlay');
        if (overlay) {
            overlay.style.display = this.container.classList.contains('mobile-open') ? 'block' : 'none';
        }
    }
}
