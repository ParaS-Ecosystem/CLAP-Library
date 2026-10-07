/*#include <dnn.hh>
#include <iostream>
#include <vector>

int main() {
    auto backend = clap::createDnnBackend();
    clap::TensorDesc dy_desc({1, 2, 2, 2});
    clap::TensorDesc db_desc({2});
    std::vector<float> dy = {1,2,3,4, 5,6,7,8};
    std::vector<float> db(2, 0.0f);
    backend->convolutionBackwardBias(dy_desc, dy.data(), db_desc, db.data());
    std::cout << "Backend: " << backend->name() << "\ndbias: ";
    for (float v : db) std::cout << v << ' ';
    std::cout << '\n';
}
*/

#include <dnn.hh>

#include <iostream>
#include <vector>


int main()
{
    auto backend = clap::createDnnBackend();


    /*
     * dy shape:
     * N = 1
     * C = 2
     * H = 2
     * W = 2
     */
    clap::TensorDesc dy_desc({
        1,
        2,
        2,
        2
    });


    /*
     * Bias gradient shape.
     *
     * CLAP_DNN4 currently expects 4D descriptors,
     * so use:
     *
     * N = 1
     * C = 2
     * H = 1
     * W = 1
     */
    clap::TensorDesc db_desc({
        1,
        2,
        1,
        1
    });


    std::vector<float> dy = {
        1, 2, 3, 4,
        5, 6, 7, 8
    };


    std::vector<float> db(
        2,
        0.0f
    );


    backend->convolutionBackwardBias(
        dy_desc,
        dy.data(),

        db_desc,
        db.data()
    );


    std::cout
        << "Backend: "
        << backend->name()
        << "\n";


    std::cout
        << "dbias: ";


    for (float v : db)
    {
        std::cout
            << v
            << ' ';
    }


    std::cout << '\n';


    return 0;
}
