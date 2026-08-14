#include <blas.hh>
#include <cstdio>

using namespace clap;

int main() {
    int n = 5;
    double x[] = {1.0, -7.0, 3.0, 9.0, -4.0};
    int incx = 1;

    auto backend = clap::BlasFactory::create();

    auto index = backend->iamax(n, x, incx);

    printf("IAMAX index = %d\n", (int)index);
    printf("Value = %lf\n", x[index]);

    return 0;
}
