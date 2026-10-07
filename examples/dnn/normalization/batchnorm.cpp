#include <dnn.hh>
#include <iostream>
#include <vector>
int main(){auto d=clap::createDnnBackend();clap::TensorDesc td({1,2,2,2});clap::BatchNormDesc bn;std::vector<float>x{1,2,3,4,5,6,7,8},y(8),scale{1,1},bias{0,0},rm(2),rv(2),sm(2),sv(2);d->batchNormForwardTraining(bn,td,x.data(),scale.data(),bias.data(),rm.data(),rv.data(),sm.data(),sv.data(),td,y.data());std::cout<<"Backend: "<<d->name()<<"\n";for(float v:y)std::cout<<v<<' ';std::cout<<'\n';}
