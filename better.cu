#include <iostream>
#include <math.h>

#define SIZE 10000

//kernel function to perform prefix sum with multiple threads
__global__
void prefixsum(int *source, int *output){
        //get index of current thread
        int xindex = threadIdx.x;
        int index = blockIdx.x * blockDim.x + threadIdx.x;

        __shared__ int destination[SIZE];
     
        if (index < SIZE) {
        	destination[xindex] = source[index];
        }

        for (int stride=1; stride<blockDim.x;stride++){
        	__syncthreads();
        	int sum;
        	if (xindex >= stride){
        	sum = source[xindex-stride];
			}
			__syncthreads();
			if (threadIdx.x >= stride){
				destination[xindex] += sum;
			}	
        	
        }

		//copy destination contents to global memory
        output[index] = destination[xindex];
        

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
		cudaFree(input);
		cudaFree(output);

}
