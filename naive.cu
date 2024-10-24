#include <iostream>
#include <math.h>
#include <sys/time.h>

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
	int *times;
  	cudaMallocManaged(&times, SIZE*sizeof(double));


	//calibrate clock
	double t0 = get_clock();
  	for (int i=0; i<SIZE; i++){
            times[i] = get_clock();
  	}
  	double t1 = get_clock();
  	printf("time per call: %f nx\n", (1000000000.0 * (t1-t0)/SIZE));		

    cudaMallocManaged(&input, sizeof(int) * SIZE);
    cudaMallocManaged(&output, sizeof(int) * SIZE);

    //initialize input array
    for (int i=0; i<SIZE; i++) {
    	input[i] = 1;
    }

	//start timer
	double start = get_clock();
        //run kernel on elements on the GPU
        prefixsum<<<1,SIZE>>>(input, output);
    //get middle time
	double middle = get_clock();


    cudaDeviceSynchronize();

    //get end time
	double end = get_clock();
	//print any errors
    printf("%s\n", cudaGetErrorString(cudaGetLastError()));

    for (int i=0;i<SIZE;i++){
         printf("%d ", output[i]);
    }
    printf("\n");

    //print clock times
    printf("start: %f, middle: %f, end: %f\n", start, middle, end);
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
