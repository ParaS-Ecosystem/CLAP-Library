#include <blas.hh>
#include <chrono>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <ctime>

using namespace clap;
using namespace std::chrono;

int main() {
    int m, n, k;

    printf("Enter m : ");
    scanf("%d", &m);

    printf("Enter n : ");
    scanf("%d", &n);

    printf("Enter k : ");
    scanf("%d", &k);

    using complex_t = std::complex<double>;

    const complex_t alpha(1.0, 0.0);
    const complex_t beta(0.0, 0.0);

    srand(time(NULL));

    complex_t *A = (complex_t *)malloc(m * k * sizeof(complex_t));
    complex_t *B = (complex_t *)malloc(k * n * sizeof(complex_t));
    complex_t *C = (complex_t *)malloc(m * n * sizeof(complex_t));

    if (A == NULL || B == NULL || C == NULL) {
        printf("malloc failed\n");
        return 1;
    }

    for (int i = 0; i < m * k; i++) {
        double real = (double)((rand() % 10) + 1);
        double imag = (double)((rand() % 10) + 1);
        A[i] = complex_t(real, imag);
    }

    for (int i = 0; i < k * n; i++) {
        double real = (double)((rand() % 10) + 1);
        double imag = (double)((rand() % 10) + 1);
        B[i] = complex_t(real, imag);
    }

    for (int i = 0; i < m * n; i++) {
        C[i] = complex_t(0.0, 0.0);
    }

    auto backend = clap::BlasFactory::create();

    auto start = high_resolution_clock::now();

    backend->gemm(
        Layout::ColMajor,
        Transpose::NoTrans,
        Transpose::NoTrans,
        m,
	n,
	k,
	alpha,
	A,
	m,
	B,
	k,
	beta,
	C,
	m
    );

    auto stop = high_resolution_clock::now();

    printf("\nResult Matrix C (%d x %d):\n", m, n);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            complex_t value = C[i + j * m];

            printf("(%10.2lf,%10.2lf) ",
                   value.real(),
                   value.imag());
        }
	printf("\n");
    }

    double total_seconds = duration<double>(stop - start).count();

    printf("\nTotal time : %.6f s\n", total_seconds);

    free(A);
    free(B);
    free(C);

    return 0;
}

