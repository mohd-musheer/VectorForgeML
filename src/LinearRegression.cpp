#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <R_ext/BLAS.h>

using namespace Rcpp;
using namespace std;

struct LinearModel {
    vector<double> coef;
    double intercept = 0.0;
};

static vector<double> solveCholesky(vector<double> A, vector<double> b, int n) {
    for (int j = 0; j < n; ++j) {
        double d = A[j + j * n];
        for (int k = 0; k < j; ++k) {
            const double ljk = A[j + k * n];
            d -= ljk * ljk;
        }
        if (d <= 1e-12) {
            d = 1e-12;
        }
        A[j + j * n] = std::sqrt(d);

        for (int i = j + 1; i < n; ++i) {
            double s = A[i + j * n];
            for (int k = 0; k < j; ++k) {
                s -= A[i + k * n] * A[j + k * n];
            }
            A[i + j * n] = s / A[j + j * n];
        }
    }

    for (int i = 0; i < n; ++i) {
        double s = b[i];
        for (int k = 0; k < i; ++k) {
            s -= A[i + k * n] * b[k];
        }
        b[i] = s / A[i + i * n];
    }

    for (int i = n - 1; i >= 0; --i) {
        double s = b[i];
        for (int k = i + 1; k < n; ++k) {
            s -= A[k + i * n] * b[k];
        }
        b[i] = s / A[i + i * n];
    }
    return b;
}

static vector<double> fitCoefNoIntercept(const NumericMatrix& X, const NumericVector& y) {
    const int n = X.nrow();
    const int p = X.ncol();
    if (y.size() != n) {
        stop("X and y must have the same number of rows");
    }
    if (p == 0) {
        return {};
    }

    vector<double> XtX(static_cast<size_t>(p) * p, 0.0);
    const char* trans = "T";
    const char* notrans = "N";
    const double one = 1.0;
    const double zero = 0.0;
    F77_CALL(dgemm)(
        trans, notrans, &p, &p, &n,
        &one, X.begin(), &n, X.begin(), &n, &zero, XtX.data(), &p
        FCONE FCONE
    );

    vector<double> Xty(p, 0.0);
    const int inc = 1;
    F77_CALL(dgemv)(
        trans, &n, &p, &one,
        X.begin(), &n, y.begin(), &inc, &zero, Xty.data(), &inc
        FCONE
    );

    for (int j = 0; j < p; ++j) {
        XtX[j + j * p] += 1e-8;
    }
    return solveCholesky(std::move(XtX), std::move(Xty), p);
}

// [[Rcpp::export]]
SEXP lr_create() {
    XPtr<LinearModel> model(new LinearModel(), true);
    return model;
}

// [[Rcpp::export]]
void lr_fit(SEXP ptr, NumericMatrix X, NumericVector y) {
    XPtr<LinearModel> model(ptr);
    const int n = X.nrow();
    const int p = X.ncol();
    if (y.size() != n) {
        stop("X and y must have the same number of rows");
    }
    if (n == 0) {
        stop("X must have at least one row");
    }

    vector<double> meanX(p, 0.0);
    for (int j = 0; j < p; ++j) {
        meanX[j] = mean(X(_, j));
    }
    const double meanY = mean(y);

    NumericMatrix Xc(n, p);
    NumericVector yc(n);
    for (int j = 0; j < p; ++j) {
        const double mj = meanX[j];
        for (int i = 0; i < n; ++i) {
            Xc(i, j) = X(i, j) - mj;
        }
    }
    for (int i = 0; i < n; ++i) {
        yc[i] = y[i] - meanY;
    }

    model->coef = fitCoefNoIntercept(Xc, yc);
    model->intercept = meanY;
    for (int j = 0; j < p; ++j) {
        model->intercept -= model->coef[j] * meanX[j];
    }
}

// [[Rcpp::export]]
NumericVector lr_predict(SEXP ptr, NumericMatrix X) {
    XPtr<LinearModel> model(ptr);
    const int n = X.nrow();
    const int p = X.ncol();
    if (static_cast<int>(model->coef.size()) != p) {
        stop("Input feature count does not match trained model");
    }

    NumericVector pred(n, model->intercept);
    const char* notrans = "N";
    const double one = 1.0;
    const int inc = 1;
    F77_CALL(dgemv)(
        notrans, &n, &p, &one,
        X.begin(), &n, model->coef.data(), &inc, &one, pred.begin(), &inc
        FCONE
    );
    return pred;
}

// [[Rcpp::export]]
NumericVector fastLm(NumericMatrix X, NumericVector y) {
    const vector<double> coef = fitCoefNoIntercept(X, y);
    return NumericVector(coef.begin(), coef.end());
}
