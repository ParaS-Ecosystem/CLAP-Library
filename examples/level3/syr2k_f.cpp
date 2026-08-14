#include <blas.hh>
#include <cstdio>

using namespace clap;

int main() {
    int n = 3;
    int k = 2;

    float alpha = 1.0f;
    float beta = 0.0f;

    // A is n x k = 3 x 2
    float A[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };

    // B is n x k = 3 x 2
    float B[] = {
        2.0f, 1.0f, 3.0f,
        1.0f, 2.0f, 4.0f
    };

    float C[] = {
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f
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
            printf("%8.2f ", C[i + j * ldc]);
        }
	printf("\n");
    }

    return 0;
}

