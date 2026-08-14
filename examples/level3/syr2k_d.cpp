
#include <blas.hh>
#include <cstdio>

using namespace clap;

int main() {
    int n = 3;
    int k = 2;

    double alpha = 1.0;
    double beta = 0.0;

    double A[] = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0
    };

    double B[] = {
        2.0, 1.0, 3.0,
        1.0, 2.0, 4.0
    };

    double C[] = {
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0
    };

    int lda = n;
    int ldb = n;
    int ldc = n;

    auto backend = clap::BlasFactory::create();

    backend->syr2k(
        Layout::ColMajor,
        Uplo::Upper,
        Transpose::NoTrans,
        n,
	k,
	alpha,
	A,
	lda,
	B,
	ldb,
	beta,
	C,
	ldc
    );

    printf("Result matrix C:\n");

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%8.2lf ", C[i + j * ldc]);
        }
	printf("\n");
    }

    return 0;
}


