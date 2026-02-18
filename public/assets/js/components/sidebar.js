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
        const algoLinks = this.algorithms.map(algo => {
            const isActive = path.includes(algo) ? 'active' : '';
            return `<a href="/docs/${algo}.html" class="sidebar-link ${isActive}">${algo}</a>`;
        }).join('');

        this.container.innerHTML = `
            <div class="sidebar-group">
                <h3>Getting Started</h3>
                <a href="/docs/index.html" class="sidebar-link ${path.endsWith('docs/index.html') || path.endsWith('docs/') ? 'active' : ''}">Introduction</a>
                <a href="/install/index.html" class="sidebar-link">Installation</a>
            </div>
            <div class="sidebar-group">
                <h3>Algorithms</h3>
                ${algoLinks}
            </div>
        `;
    }
}
