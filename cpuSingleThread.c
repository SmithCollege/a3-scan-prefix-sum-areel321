#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#define SIZE 1024

double get_clock() {
	struct timeval tv; int ok;
	ok = gettimeofday(&tv, (void *) 0);
	if (ok<0){
		printf("gettimeofday error\n");
	}
	return (tv.tv_sec*1.0+tv.tv_usec*1.0E-6);
}


int main() {
        //output: sum of all prev elements

        //allocate memory
        int* input = malloc(sizeof(int) * SIZE);
        int* output = malloc(sizeof(int) * SIZE);
        double *times = malloc(sizeof(double) * SIZE);

        double t0 = get_clock();
        for (int i=0; i<SIZE; i++){
        	times[i] = get_clock();
        }
        double t1 = get_clock();
        printf("time per call: %f nx\n", (1000000000.0 * (t1-t0)/SIZE));
        

        //initialize inputs
        for (int i=0; i<SIZE; i++) {
                input[i]=1;
        }

		double start = get_clock();
        //do the scan
        for (int i=0; i<SIZE;i++) {
                int value = 0;
                for (int j=0; j<=i; j++){
                        value += input[j];
                }
                output[i] = value;
        }
        double end = get_clock();

        //check results
        /*for (int i=0; i<SIZE; i++) {
                printf("%d ", output[i]);
        }
        printf("\n");*/

        printf("start: %f, end: %f\n", start, end);

        

        //free mem
        free(input);
        free(output);
        free(times);

        return 0;
}
