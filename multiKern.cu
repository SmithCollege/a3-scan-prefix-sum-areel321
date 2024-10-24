#include <iostream>
#include <math.h>
#include <sys/time.h>

#define SIZE 128

//kernel function to perform prefix sum with multiple threads
__global__
void prefixsum(int *input, int *output, int stride){
        //get index of current thread
        int xindex = threadIdx.x;
        //if we are outside the stride, perform the offser
		if ( xindex >= stride ) {
			output[xindex] = input[xindex] + input[xindex-stride];
		}
		//if we are in the stride, copy the value over
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

		//calibrate clock
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

		//get pointers for swapping
		source = &input[0];
		destination = &output[0];
		
        //start clock
		double start = get_clock();
		//start loop to go over arr, stride x2 each time
		for (int stride=1; stride < SIZE; stride*=2){
				//run kernel on elements on the GPU
				prefixsum<<<1,SIZE>>>(source, destination, stride);

	        	//switch pointers
	        	temp = destination;
	        	destination = source;
	        	source = temp;
        }

       	//get middle time
        double middle = get_clock();
        cudaDeviceSynchronize();
        //get end time
		double end = get_clock();

		//print errors
        printf("%s\n", cudaGetErrorString(cudaGetLastError()));

		//check for correctness
        for (int i=0;i<SIZE;i++){
                printf("%d ", output[i]);
        }
        printf("\n");
        
		//print clock times
        printf("start: %f, middle: %f, end: %f", start, middle, end);
        //check for errors
//      float maxError = 0;
//      for (int i=0;i<SIZE;i++){
//              maxError = fmax(maxError, fabs(output[i]-7));
//      }
//      std::cout << "Max error: " <<< maxError <<< std::endl;
		//free mem
		cudaFree(input);
		cudaFree(output);
		cudaFree(times);
		

}
