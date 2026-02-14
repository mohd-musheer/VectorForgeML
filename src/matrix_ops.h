#pragma once
#include "FastMatrix.h"
#include <vector>
using namespace std;

FastMatrix transpose(const FastMatrix& A);
FastMatrix matmul(const FastMatrix& A,const FastMatrix& B);
vector<double> matvec(const FastMatrix& A,const vector<double>& x);
vector<double> choleskySolve(FastMatrix A, vector<double> b);
