#include "matrix_ops.h"
#include <cblas.h>
#include <cmath>
using namespace std;


// transpose
FastMatrix transpose(const FastMatrix& A){
    FastMatrix T(A.cols,A.rows);

    #pragma omp parallel for
    for(int i=0;i<A.rows;i++)
        for(int j=0;j<A.cols;j++)
            T(j,i)=A(i,j);

    return T;
}


// BLAS matrix multiply
FastMatrix matmul(const FastMatrix& A,const FastMatrix& B){

    FastMatrix C(A.rows,B.cols);

    cblas_dgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        A.rows,
        B.cols,
        A.cols,
        1.0,
        A.ptr(),
        A.cols,
        B.ptr(),
        B.cols,
        0.0,
        C.ptr(),
        C.cols
    );

    return C;
}


// matrix vector
vector<double> matvec(const FastMatrix& A,const vector<double>& x){

    vector<double> y(A.rows,0);

    cblas_dgemv(
        CblasRowMajor,
        CblasNoTrans,
        A.rows,
        A.cols,
        1.0,
        A.ptr(),
        A.cols,
        x.data(),
        1,
        0.0,
        y.data(),
        1
    );

    return y;
}



// Cholesky solver (fast + stable)
vector<double> choleskySolve(FastMatrix A, vector<double> b){

    int n=A.rows;

    // decomposition
    for(int i=0;i<n;i++){
        for(int j=0;j<=i;j++){
            double sum=A(i,j);

            for(int k=0;k<j;k++)
                sum-=A(i,k)*A(j,k);

            if(i==j)
                A(i,j)=sqrt(sum);
            else
                A(i,j)=sum/A(j,j);
        }
    }

    // forward
    for(int i=0;i<n;i++){
        for(int k=0;k<i;k++)
            b[i]-=A(i,k)*b[k];
        b[i]/=A(i,i);
    }

    // backward
    for(int i=n-1;i>=0;i--){
        for(int k=i+1;k<n;k++)
            b[i]-=A(k,i)*b[k];
        b[i]/=A(i,i);
    }

    return b;
}
