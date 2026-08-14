#include <blas.hh>
#include <cstdio>

using namespace clap;

int main() {
    int n = 5;
    float x[] = {1.0f, -7.0f, 3.0f, 9.0f, -4.0f};
    int incx = 1;

    auto backend = clap::BlasFactory::create();

    auto index = backend->iamax(n, x, incx);

    printf("IAMAX index = %d\n", (int)index);
    printf("Value = %f\n", x[index]);

    return 0;
}

