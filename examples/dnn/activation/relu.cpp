#include <dnn.hh>
#include <iostream>
#include <vector>
int main(){auto d=clap::createDnnBackend();clap::TensorDesc td({1,1,2,4});clap::ActivationDesc a;std::vector<float>x{-3,-1,0,1,2,-2,5,6},y(8);d->activationForward(a,td,x.data(),td,y.data());std::cout<<"Backend: "<<d->name()<<"\n";for(float v:y)std::cout<<v<<' ';std::cout<<'\n';}
