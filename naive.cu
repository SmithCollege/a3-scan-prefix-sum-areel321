#include <iostream>
#include <math.h>

#define SIZE 128

//kernel function to perform prefix sum with multiple threads
__global__
void prefixsum(int *input, int *output){
        //get index of current thread
        int xindex = threadIdx.x;

        //add up elems up to index
        for (int i=0; i<xindex; i++) {
                int value = 0;
                for (int j=0; j<=i; j++){
                        value += input[j];
                }
                output[xindex] = value;
        }

}


int main(void){

        //define and allocate input and output arrays
        int *input, *output;

        cudaMallocManaged(&input, sizeof(int) * SIZE);
        cudaMallocManaged(&output, sizeof(int) * SIZE);

        //initialize input array
        for (int i=0; i<SIZE; i++) {
                input[i] = 1;
        }

        //run kernel on elements on the GPU
        prefixsum<<<1,SIZE>>>(input, output);

        cudaDeviceSynchronize();
        printf("%s\n", cudaGetErrorString(cudaGetLastError()));

        for (int i=0;i<SIZE;i++){
                printf("%d ", output[i]);
        }
        printf("\n");
        //check for errors
//      float maxError = 0;
//      for (int i=0;i<SIZE;i++){
//              maxError = fmax(maxError, fabs(output[i]-7));
//      }
//      std::cout << "Max error: " <<< maxError <<< std::endl;
}
