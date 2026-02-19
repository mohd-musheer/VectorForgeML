export class Sidebar {
    constructor() {
        this.container = document.getElementById('sidebar');
        this.algorithms = [
            "accuracy_score", "ColumnTransformer", "confusion_matrix", "confusion_stats", "DecisionTree",
            "drop_constant_columns", "f1_score", "find_best_k", "fit_linear_model", "KMeans", "KNN",
            "LabelEncoder", "LinearRegression", "LogisticRegression", "macro_f1", "macro_precision",
            "macro_recall", "MinMaxScaler", "mse", "OneHotEncoder", "PCA", "Pipeline",
            "plot_confusion_matrix", "precision_score", "predict_linear_model", "r2_score", "RandomForest",
            "recall_score", "RidgeRegression", "rmse", "SoftmaxRegression", "StandardScaler", "train_test_split"
        ];
    }

    render() {
        if (!this.container) return;

        const path = window.location.pathname;

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

        let sidebarHTML = `
            <div class="sidebar-group">
                <h3>Getting Started</h3>
                <a href="/docs/index.html" class="sidebar-link ${path.endsWith('docs/index.html') || path.endsWith('docs/') ? 'active' : ''}">Introduction</a>
                <a href="/install/index.html" class="sidebar-link">Installation</a>
            </div>
        `;

        for (const [groupName, items] of Object.entries(groups)) {
            const links = items.map(algo => {
                const isActive = path.includes(`/${algo}.html`) ? 'active' : '';
                return `<a href="/docs/${algo}.html" class="sidebar-link ${isActive}">${algo}</a>`;
            }).join('');

            sidebarHTML += `
                <div class="sidebar-group">
                    <h3>${groupName}</h3>
                    ${links}
                </div>
            `;
        }

        this.container.innerHTML = sidebarHTML;
    }
}
