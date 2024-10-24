#include <iostream>
#include <math.h>
#include <sys/time.h>

#define SIZE 1024

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


double get_clock() {
        struct timeval tv; int ok;
        ok = gettimeofday(&tv, (void *) 0);
        if (ok<0){
                printf("gettimeofday error");
        }
        return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}

int main(void){

        //define and allocate input and output arrays
        int *input, *output;
        int* source;
        int* destination;
        int* temp;
		int *times;
        cudaMallocManaged(&times, SIZE*sizeof(double));
        cudaMallocManaged(&input, sizeof(int) * SIZE);
        cudaMallocManaged(&output, sizeof(int) * SIZE);


	
        double t0 = get_clock();
        for (int i=0; i<SIZE; i++){
            times[i] = get_clock();
        }
        double t1 = get_clock();
        printf("time per call: %f nx\n", (1000000000.0 * (t1-t0)/SIZE));
		
        //initialize input array
        for (int i=0; i<SIZE; i++) {
                input[i] = 1;
        }


		source = &input[0];
		destination = &output[0];
        
        

        //run kernel on elements on the GPU
        
        	//call kernel to sync threads
        	//int stride = 1;
		double start = get_clock();
			for (int stride=1; stride < SIZE; stride*=2){
				//stride *=2;
				
				prefixsum<<<1,SIZE>>>(source, destination, stride);
			

	        	//switch pointers
	        	temp = destination;
	        	destination = source;
	        	source = temp;


	        	
        	}
        	
        	double middle = get_clock();
       

        cudaDeviceSynchronize();
	double end = get_clock();

         printf("start: %f, middle: %f, end: %f", start, middle\
, end);
        printf("%s\n", cudaGetErrorString(cudaGetLastError()));

        /*for (int i=0;i<SIZE;i++){
                printf("%d ", output[i]);
        }
        printf("\n");*/
        //check for errors
//      float maxError = 0;
//      for (int i=0;i<SIZE;i++){
//              maxError = fmax(maxError, fabs(output[i]-7));
//      }
//      std::cout << "Max error: " <<< maxError <<< std::endl;
		cudaFree(input);
		cudaFree(output);
		cudaFree(times);

}
