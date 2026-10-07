#include <dnn.hh>
#include <iostream>
#include <vector>
int main(){auto b=clap::createDnnBackend();clap::TensorDesc d({1,1,1,4});std::vector<float>a{1,2,3,4},c{4,3,2,1},y(4);b->tensorAdd(d,a.data(),d,c.data(),d,y.data());std::cout<<"Backend: "<<b->name()<<"\n";for(float v:y)std::cout<<v<<' ';std::cout<<'\n';}
