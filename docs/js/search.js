const algorithms = [
    "accuracy_score", "ColumnTransformer", "confusion_matrix", "confusion_stats", "DecisionTree",
    "drop_constant_columns", "f1_score", "find_best_k", "fit_linear_model", "KMeans", "KNN",
    "LabelEncoder", "LinearRegression", "LogisticRegression", "macro_f1", "macro_precision",
    "macro_recall", "MinMaxScaler", "mse", "OneHotEncoder", "PCA", "Pipeline",
    "plot_confusion_matrix", "precision_score", "predict_linear_model", "r2_score", "RandomForest",
    "recall_score", "RidgeRegression", "rmse", "SoftmaxRegression", "StandardScaler", "train_test_split"
];

export function initSearch() {
    const searchInput = document.getElementById('global-search');
    const resultsContainer = document.getElementById('search-results');

    if (!searchInput || !resultsContainer) {
        console.warn('Search elements not found');
        return;
    }

    function doSearch(query) {
        if (!query) {
            resultsContainer.style.display = 'none';
            return;
        }

        const lowerQuery = query.toLowerCase();
        const matches = algorithms.filter(algo => algo.toLowerCase().includes(lowerQuery));

        if (matches.length > 0) {
            resultsContainer.innerHTML = matches.map((algo, index) => `
                <a href="/docs/${algo}.html" class="search-result-item" data-index="${index}" style="
                    display: block;
                    padding: 0.75rem 1rem;
                    text-decoration: none;
                    color: var(--text-secondary);
                    border-bottom: 1px solid var(--glass-border);
                    transition: background 0.2s;
                " onmouseover="this.style.background='rgba(255,255,255,0.05)'" onmouseout="this.style.background='transparent'">
                    <span style="color: var(--accent-primary); font-weight: 600;">${algo}</span>
                </a>
            `).join('');
            resultsContainer.style.display = 'block';
        } else {
            resultsContainer.innerHTML = `<div style="padding: 1rem; color: var(--text-secondary); font-size: 0.9rem;">No results found</div>`;
            resultsContainer.style.display = 'block';
        }
    }

    // Event Listeners
    searchInput.addEventListener('input', (e) => {
        doSearch(e.target.value);
    });

    searchInput.addEventListener('focus', () => {
        if (searchInput.value) doSearch(searchInput.value);
    });

    // Enter key navigation
    searchInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            const firstResult = resultsContainer.querySelector('a');
            if (firstResult) {
                window.location.href = firstResult.href;
            }
        }
    });

    // Close on click outside
    document.addEventListener('click', (e) => {
        if (!searchInput.contains(e.target) && !resultsContainer.contains(e.target)) {
            resultsContainer.style.display = 'none';
        }
    });
}
