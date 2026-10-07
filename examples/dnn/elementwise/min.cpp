#include <dnn.hh>
#include <iostream>
#include <vector>
int main(){auto b=clap::createDnnBackend();clap::TensorDesc d({1,1,1,4});std::vector<float>a{1,5,3,8},c{4,3,6,2},y(4);b->tensorMin(d,a.data(),d,c.data(),d,y.data());std::cout<<"Backend: "<<b->name()<<"\n";for(float v:y)std::cout<<v<<' ';std::cout<<'\n';}
