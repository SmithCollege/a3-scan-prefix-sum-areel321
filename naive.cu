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


	double start = get_clock();
        //run kernel on elements on the GPU
        prefixsum<<<1,SIZE>>>(input, output);
	double middle = get_clock();


        cudaDeviceSynchronize();
	double end = get_clock();
        printf("%s\n", cudaGetErrorString(cudaGetLastError()));

	 printf("start: %f, middle: %f, end: %f", start, middle, end);

       /* for (int i=0;i<SIZE;i++){
                printf("%d ", output[i]);
        }
        printf("\n");*/
        //check for errors
//      float maxError = 0;
//      for (int i=0;i<SIZE;i++){
//              maxError = fmax(maxError, fabs(output[i]-7));
//      }
//      std::cout << "Max error: " <<< maxError <<< std::endl;
}
