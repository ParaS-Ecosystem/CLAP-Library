#include <blas.hh>
#include <cstdio>

using namespace clap;

int main() {
    int n = 4;

    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};

    int incx = 1;

    auto backend = clap::BlasFactory::create();

    float result = backend->nrm2(n, x, incx);

    printf("NRM2 result = %f\n", result);

    return 0;

