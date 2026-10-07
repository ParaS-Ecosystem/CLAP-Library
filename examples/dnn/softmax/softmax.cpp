#include <dnn.hh>
#include <iostream>
#include <vector>
int main(){auto d=clap::createDnnBackend();clap::TensorDesc td({1,3,1,1});std::vector<float>x{1,2,3},y(3);d->softmaxForward(td,x.data(),td,y.data());std::cout<<"Backend: "<<d->name()<<"\n";for(float v:y)std::cout<<v<<' ';std::cout<<'\n';}
