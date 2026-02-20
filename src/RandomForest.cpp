#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <random>
#include <map>
#include <numeric>

using namespace Rcpp;
using std::vector;

struct Node{
  int feature;
  double thresh;
  double value;
  Node* left;
  Node* right;
};

double majority(const NumericVector& y, const vector<int>& idx){
  std::map<double,int> c;
  for(int i : idx) c[y[i]]++;
  return std::max_element(c.begin(),c.end(),
        [](auto&a,auto&b){return a.second<b.second;})->first;
}

double mean_val(const NumericVector& y, const vector<int>& idx){
  double s=0;
  for(int i : idx) s+=y[i];
  return s/idx.size();
}

Node* build(const NumericMatrix& X, const NumericVector& y,
            vector<int>& idx, int depth, int max_depth,
            int mtry, bool cls, std::mt19937& rng){

  int n = idx.size();
  int p = X.ncol();

  Node* node = new Node();

  bool pure = true;
  for(int i=1; i<n; i++){
    if(y[idx[i]] != y[idx[0]]){
      pure = false;
      break;
    }
  }

  if(depth >= max_depth || n <= 2 || pure){
    node->value = cls ? majority(y, idx) : mean_val(y, idx);
    node->left = node->right = NULL;
    return node;
  }

  vector<int> feats(p);
  std::iota(feats.begin(), feats.end(), 0);
  std::shuffle(feats.begin(), feats.end(), rng);
  if (mtry < p) feats.resize(mtry);

  double best_score = -1.0;
  int best_feature = -1;
  double best_thresh = 0.0;

  double sum_total = 0;
  std::map<double, int> class_counts_total;
  if (!cls) {
    for (int i : idx) sum_total += y[i];
  } else {
    for (int i : idx) class_counts_total[y[i]]++;
  }

  vector<int> sorted_idx = idx;

  for(int f : feats){
    std::sort(sorted_idx.begin(), sorted_idx.end(), [&](int a, int b){
        return X(a, f) < X(b, f);
    });

    double sum_L = 0;
    std::map<double, int> class_counts_L;

    for(int i = 0; i < n - 1; i++){
      int id = sorted_idx[i];

      if (!cls) {
        sum_L += y[id];
      } else {
        class_counts_L[y[id]]++;
      }

      if (X(sorted_idx[i], f) < X(sorted_idx[i+1], f)) {
        double IL = 0, IR = 0;
        double nL = i + 1;
        double nR = n - nL;

        if (!cls) {
          double sum_R = sum_total - sum_L;
          IL = (sum_L * sum_L) / nL;
          IR = (sum_R * sum_R) / nR;
        } else {
          for (auto const& [val, count_total] : class_counts_total) {
            double cL = class_counts_L[val];
            double cR = count_total - cL;
            if(nL > 0) IL += (cL * cL) / nL;
            if(nR > 0) IR += (cR * cR) / nR;
          }
        }

        double score = IL + IR;
        if (score > best_score) {
          best_score = score;
          best_feature = f;
          best_thresh = (X(sorted_idx[i], f) + X(sorted_idx[i+1], f)) / 2.0;
        }
      }
    }
  }

  if(best_feature == -1){
    node->value = cls ? majority(y, idx) : mean_val(y, idx);
    node->left = node->right = NULL;
    return node;
  }

  node->feature = best_feature;
  node->thresh = best_thresh;

  vector<int> li, ri;
  li.reserve(n); ri.reserve(n);
  for(int i : idx) {
    if(X(i, best_feature) <= best_thresh) li.push_back(i);
    else ri.push_back(i);
  }

  node->left = build(X, y, li, depth+1, max_depth, mtry, cls, rng);
  node->right = build(X, y, ri, depth+1, max_depth, mtry, cls, rng);

  return node;
}

double predict_tree(Node* n, const NumericVector& x){
  if(!n->left) return n->value;
  return x[n->feature] <= n->thresh ?
    predict_tree(n->left, x) :
    predict_tree(n->right, x);
}

struct Forest{
  vector<Node*> trees;
  int max_depth;
  int mtry;
  bool classification;
};

// [[Rcpp::export]]
SEXP rf_create(int ntrees,int depth,int mtry,bool cls){
  XPtr<Forest> f(new Forest(),true);
  f->trees.resize(ntrees);
  f->max_depth=depth;
  f->mtry=mtry;
  f->classification=cls;
  return f;
}

// [[Rcpp::export]]
void rf_fit(SEXP ptr, NumericMatrix X, NumericVector y){

  XPtr<Forest> f(ptr);

  int n=X.nrow();
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(0,n-1);

  for(size_t t=0;t<f->trees.size();t++){
    vector<int> boot_idx(n);
    for(int i=0;i<n;i++) boot_idx[i] = dist(rng);

    f->trees[t] = build(X, y, boot_idx, 0, f->max_depth, f->mtry, f->classification, rng);
  }
}

// [[Rcpp::export]]
NumericVector rf_predict(SEXP ptr, NumericMatrix X){

  XPtr<Forest> f(ptr);

  int n=X.nrow();
  NumericVector out(n);

  for(int i=0;i<n;i++){

    vector<double> preds;

    NumericVector xi = X(i,_);
    for(Node* t:f->trees)
      preds.push_back(predict_tree(t, xi));

    if(f->classification){
      std::map<double,int> c;
      for(double v:preds) c[v]++;
      out[i]=std::max_element(c.begin(),c.end(),
          [](auto&a,auto&b){return a.second<b.second;})->first;
    }
    else{
      out[i]=std::accumulate(preds.begin(),preds.end(),0.0)/preds.size();
    }
  }

  return out;
}
