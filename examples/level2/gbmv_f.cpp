#include <blas.hh>
#include <cstdio>

using namespace clap;

int main() {
    int m = 4;
    int n = 4;

    int kl = 1;
    int ku = 1;

    float alpha = 1.0f;
    float beta = 0.0f;

    /*
        Logical matrix:

        1  2  0  0
        3  4  5  0
        0  6  7  8
        0  0  9 10
    */

    int lda = kl + ku + 1;

    float A[] = {
        0, 1, 3,
        2, 4, 6,
        5, 7, 9,
        8,10, 0
    };

    float x[] = {
        1.0f,
	1.0f,
	1.0f,
	1.0f
    };

    float y[] = {
        0.0f,
	0.0f,
	0.0f,
	0.0f
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
        printf("%8.2f\n", y[i]);
    }

    return 0;
}
