#include <blas.hh>
#include <cstdio>

using namespace clap;

int main() {
    int n = 4;

    double x[] = {1.0, 2.0, 3.0, 4.0};

    int incx = 1;

    auto backend = clap::BlasFactory::create();

    double result = backend->nrm2(n, x, incx);

    printf("NRM2 result = %lf\n", result);

    return 0;
}
