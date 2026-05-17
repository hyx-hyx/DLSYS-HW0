#include <cmath>
#include <iostream>
void mat_mul(float* result,const float *X,float *theta,size_t m,size_t n,size_t k){
    for(size_t i=0;i<m;++i){
        for(size_t k_idx=0;k_idx<k;++k_idx){
            float sum=0;
            for(size_t j=0;j<n;++j){
                sum+=X[i*n+j]*theta[j*k+k_idx];
            }
            result[i*k+k_idx]=sum;
        }
    }
}
int main(){
    float x_theta[4];
    float X_batch[4]={1,2,3,4};
    float y_batch[4]={1,2,3,4};
    mat_mul(x_theta,X_batch,y_batch,2,2,2);
    std::cout<<x_theta[0]<<" "<<x_theta[1]<<" "<<x_theta[2]<<" "<<x_theta[3]<<std::endl;
}

// [1 2]  [1 2]
// [3 4]  [3 4]