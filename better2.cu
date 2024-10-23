#include <iostream>
#include <math.h>

#define SIZE 8

//kernel function to perform prefix sum with multiple threads
__global__
void prefixsum(int *input, int *output, int stride){
        //get index of current thread
        int xindex = threadIdx.x;
        //int index = blockIdx.x * blockDim.x + threadIdx.x;
		if ( xindex >= stride ) {
			output[xindex] = input[xindex] + input[xindex-stride];
		}
		if ( xindex < stride ) {
			output[xindex] = input[xindex];
		}
		
		
        

}


int main(void){

        //define and allocate input and output arrays
        int *input, *output;
        int* source;
        int* destination;
        int* temp;

        cudaMallocManaged(&input, sizeof(int) * SIZE);
        cudaMallocManaged(&output, sizeof(int) * SIZE);
		
        //initialize input array
        /*for (int i=0; i<SIZE; i++) {
                input[i] = 1;
        }*/

        input[0] = 3;
        input[1] = 1;
        input[2] = 7;
        input[3] = 0;
        input[4] = 4;
        input[5] = 1;
        input[6] = 6;
        input[7] = 3;

		source = &input[0];
		destination = &output[0];
        
        

        //run kernel on elements on the GPU
        
        	//call kernel to sync threads
        	//int stride = 1;
			for (int stride=1; stride < SIZE; stride*=2){
				//stride *=2;
				
				prefixsum<<<1,SIZE>>>(source, destination, stride);
			

	        	//switch pointers
	        	temp = destination;
	        	destination = source;
	        	source = temp;


	        	
        	}
        	
        	
       

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
