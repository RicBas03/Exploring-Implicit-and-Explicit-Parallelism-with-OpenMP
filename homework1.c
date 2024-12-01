#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <sys/time.h>
#ifdef _OPENMP
	#include <omp.h>
#endif

//---------------------------------------SYMMETRY CHECK---------------------------------------
bool checkSym(int n, float **matrix){
   //suppose the matrix is symmetric
   bool check = true;
   
   double wt1, wt2;
   
   //start the wall-clock time
   wt1=omp_get_wtime();

   //symmetry check
   for(int i=0; i<n; i++){
      for(int j=i+1; j<n; j++){ //check only the upper part of the matrix
         //set the variable to false if the matrix isn't symmetric
         if(matrix[i][j]!=matrix[j][i]){
            check = false;
         }
      }
   }
   
   //stop the wall-clock time
   wt2=omp_get_wtime();
   
   //print results
   printf("\nSYMMETRY CHECK wall-clock time = %12.4g sec\n", wt2-wt1);
   
   return check;
}

//---------------------------------------MATRIX TRANSPOSITION---------------------------------------
double avg_time_serial;

float** matTranspose(int n, float **matrix){
   
   //allocate the transpose matrix
   float **transpose = (float **)malloc(n * sizeof(float *));
   if(transpose == NULL) {
      printf("Error.\n");
      return NULL;
   }
   
   for(int i=0; i<n; i++){
      transpose[i] = (float *)malloc(n * sizeof(float));
      if(transpose[i] == NULL){
         printf("Error.\n");
         return NULL;
      }
   }
   
   double wt1, wt2;
   
   //variable for the sum of the times
   double total_time_serial = 0.0;

   //run the transposition 10 times
   for (int run = 0; run < 10; run++) {
      wt1 = omp_get_wtime();
   
      //transposition
      for(int i=0; i<n; i++){
         for(int j=0; j<n; j++){
            transpose[i][j] = matrix[j][i];
         }
      }

      wt2 = omp_get_wtime();
      
      //add the time calculated to the total variable
      double time_serial = wt2 - wt1;
      total_time_serial += time_serial;
   }

   //compute average serial time and bandwidth
   avg_time_serial = total_time_serial / 10;
   double avg_bandwidth = (2.0*n*n*sizeof(float))/avg_time_serial/1e9;

   //print results
   printf("\nMATRIX TRANSPOSITION SERIAL average time = %12.8f sec", avg_time_serial);
   printf("\nMATRIX TRANSPOSITION SERIAL bandwidth = %12.4f GB/sec\n", avg_bandwidth);
   
   return transpose;
   
}

//---------------------------------------SYMMETRY CHECK IMPLICIT PARALLELIZATION---------------------------------------
bool checkSymImp(int n, float **matrix){

   bool check = true;
   
   double wt1, wt2;
   
   wt1=omp_get_wtime();

   #pragma simd
   for(int i=0; i<n; i++){
      #pragma unroll(dynamic)
      for(int j=i+1; j<n; j++){
         if(matrix[i][j]!=matrix[j][i]){
            check = false;
         }
      }
   }
   
   wt2=omp_get_wtime();
   
   printf("\nSYMMETRY CHECK IMPLICIT PARALLELIZATION wall-clock time = %12.4g sec\n", wt2-wt1);
   
   return check;
}

//---------------------------------------MATRIX TRANSPOSITION IMPLICIT PARALLELIZATION---------------------------------------
double avg_timeImp;
float** matTransposeImp(int n, float **matrix){

   if(matrix == NULL) {
      printf("Error.\n");
      return NULL;
   }
   
   float **transpose = (float **)malloc(n * sizeof(float *));
   if(transpose == NULL) {
      printf("Error.\n");
      return NULL;
   }
   
   for(int i=0; i<n; i++){
      transpose[i] = (float *)malloc(n * sizeof(float));
      if(transpose[i] == NULL){
         printf("Error.\n");
         return NULL;
      }
   }
   
   double wt1, wt2;

   double total_time_parallel = 0.0;

   for (int run = 0; run < 10; run++) {
      wt1 = omp_get_wtime();

      //size of the block
      const int SIZE = 64;
   
      //split the code into blocks
      for(int i=0; i<n; i+=SIZE){
         for(int j=0; j<n; j+=SIZE){
            for(int ii=i; ii<i+SIZE && ii<n; ii++) {
               for (int jj=j; jj<j+SIZE && jj<n; jj++) {
                  transpose[ii][jj] = matrix[jj][ii];
               }
            }
         }
      }

      wt2 = omp_get_wtime();
      double time_parallel = wt2 - wt1;
      total_time_parallel += time_parallel;
   }

   //compute average parallel time and bandwidth
   avg_timeImp = total_time_parallel / 10;
   double avg_bandwidth = (2.0*n*n*sizeof(float))/avg_timeImp/1e9;

   //print results
   printf("\nMATRIX TRANSPOSITION IMPLICIT PARALLELIZATION average time = %12.8f sec", avg_timeImp);
   printf("\nMATRIX TRANSPOSITION IMPLICIT PARALLELIZATION bandwidth = %12.4f GB/sec\n", avg_bandwidth);
   
   return transpose;
}

//---------------------------------------SYMMETRY CHECK OPENMP---------------------------------------
bool checkSymOMP(int n, float **matrix){
   bool check = true;
   bool local_check = true;
   
   double wt1, wt2;
   
   printf("\nSYMMETRY CHECK OPENMP PARALLELIZATION\n");
   printf("\n  NTHREADS  |    TIME(s)\n");
   
   for(int num_threads = 1; num_threads <= 64; num_threads *= 2){
      omp_set_num_threads(num_threads);
   
      wt1=omp_get_wtime();
   
      #pragma omp parallel for shared(check)
      for(int i=0; i<n; i++){
         #pragma omp flush(check)
         for(int j=i+1; j<n; j++){
            if(matrix[i][j]!=matrix[j][i]){
               check = false;
            }
         }
      }
      
      wt2=omp_get_wtime();
      
      printf("%11d | %12.4f\n", num_threads, wt2-wt1);
      
   }
   
   return check;
}

//---------------------------------------MATRIX TRANSPOSITION OPENMP---------------------------------------
float** matTransposeOMP(int n, float **matrix){

   if(matrix == NULL) {
      printf("Error.\n");
      return NULL;
   }
   
   float **transpose = (float **)malloc(n * sizeof(float *));
   if(transpose == NULL) {
      printf("Error.\n");
      return NULL;
   }
   
   for(int i=0; i<n; i++){
      transpose[i] = (float *)malloc(n * sizeof(float));
      if(transpose[i] == NULL){
         printf("Error.\n");
         return NULL;
      }
   }

   double wt1, wt2;
   
   //open CSV file for writing results
   FILE *file = fopen("scaling_results.csv", "w");
   if (file == NULL) {
      perror("Failed to open file");
      return NULL;
   }

   //write header for CSV
   fprintf(file, "Num_Threads;Time;Speedup;Efficiency;Bandwidth\n");
   
   printf("\nMATRIX TRANSPOSITION OPENMP PARALLELIZATION\n");
   printf("\n  NTHREADS  |    TIME(s)   | SPEEDUP | EFFICIENCY | BANDWIDTH(GB/s)\n");
   
   //run 10 times for every number of threads (1,2,4,8,16,32,64)
   for(int num_threads = 1; num_threads <= 64; num_threads *= 2){
      omp_set_num_threads(num_threads);

      double total_time_parallel = 0.0;

      for (int run = 0; run < 10; run++){
         wt1 = omp_get_wtime();

         const int SIZE=64;
   
         #pragma omp parallel for collapse(2)
         for(int i=0; i<n; i+=SIZE){
            for(int j=0; j<n; j+=SIZE){
               for(int ii=i; ii<i+SIZE && ii<n; ii++) {
                  for (int jj=j; jj<j+SIZE && jj<n; jj++) {
                     transpose[ii][jj] = matrix[jj][ii];
                  }
               }
            }
         }

         wt2 = omp_get_wtime();
         double time_parallel = wt2 - wt1;
         total_time_parallel += time_parallel;
        }

        //compute average parallel time, speedup, efficiency, and bandwidth
        double avg_time_parallel = total_time_parallel / 10;
        double avg_speedup = avg_time_serial / avg_time_parallel;
        double avg_efficiency = avg_speedup / num_threads;
        double avg_bandwidth = (2.0*n*n*sizeof(float))/avg_time_parallel/1e9;

        //print results for each thread count
        printf("%11d | %12.8f | %7.2f | %9.2f%% | %12.4f\n", 
        num_threads, avg_time_parallel, avg_speedup, avg_efficiency * 100, avg_bandwidth);
        
        //write results to CSV
        fprintf(file, "%11d;%12.8f;%7.2f;%9.2f%%;%12.4f\n", 
        num_threads, avg_time_parallel, avg_speedup, avg_efficiency * 100, avg_bandwidth);
        
   }
   
   //close CSV file
   fclose(file);
   
   return transpose;

}
//---------------------------------------MATRIX TRANSPOSITION CHECK---------------------------------------
bool checkTrans(int n, float **matrix, float **transpose){
   bool check=true;
   
   //check if the matrix has been transposed correctly
   for(int i=0; i<n; i++){
      for(int j=0; j<n; j++){
         if(matrix[i][j]!=transpose[j][i]){
            check = false;
         }
      }
   }
   
   return check;
}
//---------------------------------------PRINT MATRIX---------------------------------------
void print(int n, float **matrix){
   for(int i=0; i<n; i++){
		for(int j=0; j<n; j++){
			printf("%f ", matrix[i][j]);
		}
		printf("\n");
   }
}
//---------------------------------------------MAIN---------------------------------------------
int main(int argc, char *argv[]) {
   
   int n = atoi(argv[1]); //converts a string passed as a command line argument into an integer 
   if(!((n>0)&&((n&(n-1))==0))) return 1; //check if n is a power of 2
   
   //allocate matrix
   float **m = (float **)malloc(n * sizeof(float *));
   if(m == NULL) {
      printf("Error.\n");
      return 1;
   }
   
   for(int i=0; i<n; i++){
      m[i] = (float *)malloc(n * sizeof(float));
      if(m[i] == NULL){
         printf("Error.\n");
         return 1;
      }
   }
   
   //allocate transpose matrix
   float **t = (float **)malloc(n * sizeof(float *));
   if(t == NULL) {
      printf("Error.\n");
      return 1;
   }
   
   for(int i=0; i<n; i++){
      t[i] = (float *)malloc(n * sizeof(float));
      if(t[i] == NULL){
         printf("Error.\n");
         return 1;
      }
   }
 
   srand(time(NULL));
 
   //INITIALIZATION
	for(int i=0; i<n; i++){
		for(int j=0; j<n; j++){
			m[i][j] = ((float)rand()/RAND_MAX)*10; //random float number from 0 to 10
		}
	}
 
   //PRINT THE MATRIX
   //print(n, m);
   
   printf("\n-----------------------------SYMMETRY CHECK-----------------------------\n");
   //CHECKSYM SERIAL
   //checkSym(n, m);
   if(checkSym(n, m)==true) printf("\nSYMMETRIC\n"); else printf("\nNOT symmetric\n");
      
   //CHECKSYM IMPLICIT PARALLELIZATION
   //checkSymImp(n, m);
   if(checkSymImp(n, m)) printf("\nSYMMETRIC\n"); else printf("\nNOT symmetric\n");
   
   //CHECKSYM OPENMP PARALLELIZATION
   //checkSymOMP(n, m);
   if(checkSymOMP(n, m)) printf("\nSYMMETRIC\n"); else printf("\nNOT symmetric\n");
   
   printf("\n--------------------------MATRIX TRANSPOSITION--------------------------\n");
      
   //MATTRANSPOSE SERIAL
   t = matTranspose(n, m);
   //if(checkTrans(n, m, t)) printf("\nCORRECT\n"); else printf("\nUNCORRECT\n"); //check if the transposition is correct
   //print(n, t);
   
   //MATTRANSPOSE IMPLICIT PARALLELIZATION
   t = matTransposeImp(n, m);
   //if(checkTrans(n, m, t)) printf("\nCORRECT\n"); else printf("\nUNCORRECT\n");
   //print(n, t);
   
   //MATTRANSPOSE OPENMP PARALLELIZATION
   t = matTransposeOMP(n, m);
   //if(checkTrans(n, m, t)) printf("\nCORRECT\n"); else printf("\nUNCORRECT\n");
   //print(n, t);
   
   //free the matrices
   for(int i=0; i<n; i++){
      free(m[i]);
      free(t[i]);
   }
   free(m);
   free(t);
 	
	return 0;
}