#include <dnn.hh>
#include <iostream>
#include <vector>
int main(){auto d=clap::createDnnBackend();clap::TensorDesc xD({1,1,4,4}),yD({1,1,2,2});clap::PoolingDesc p;std::vector<float>x(16),y(4);for(int i=0;i<16;++i)x[i]=i+1;d->poolingForward(p,xD,x.data(),yD,y.data());std::cout<<"Backend: "<<d->name()<<"\n";for(float v:y)std::cout<<v<<' ';std::cout<<'\n';}
