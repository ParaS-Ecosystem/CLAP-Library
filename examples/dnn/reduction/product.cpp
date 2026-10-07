#include <dnn.hh>
#include <iostream>
#include <vector>
int main(){auto b=clap::createDnnBackend();clap::TensorDesc x({1,1,1,4}),y({1,1,1,1});std::vector<float>a{1,2,3,4},o(1);b->tensorReduceProduct(x,a.data(),y,o.data());std::cout<<"Backend: "<<b->name()<<"\n"<<o[0]<<'\n';}
