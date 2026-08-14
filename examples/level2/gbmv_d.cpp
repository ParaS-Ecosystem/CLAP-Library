#include <blas.hh>
#include <cstdio>

using namespace clap;

int main() {
    int m = 4;
    int n = 4;

    int kl = 1;
    int ku = 1;

    double alpha = 1.0;
    double beta = 0.0;

    int lda = kl + ku + 1;

    double A[] = {
        0, 1, 3,
        2, 4, 6,
        5, 7, 9,
        8,10, 0
    };

    double x[] = {
        1.0,
	1.0,
	1.0,
	1.0
    };

    double y[] = {
        0.0,
	0.0,
	0.0,
	0.0
    };

    int incx = 1;
    int incy = 1;

    auto backend = clap::BlasFactory::create();

    backend->gbmv(
        Layout::ColMajor,
        Transpose::NoTrans,
        m,
	    n,
	    kl,
	    ku,
	    alpha,
	    A,
	    lda,
	    x,
	    incx,
	    beta,
	    y,
	    incy
    );

    printf("Result vector y:\n");

    for (int i = 0; i < m; i++) {
        printf("%8.2lf\n", y[i]);
    }

    return 0;
}



