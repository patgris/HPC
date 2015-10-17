#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 32

__global__ void sumVecReduce(int *g_idata, int *g_odata,int num_vec) {
		__shared__ int sdata[BLOCK_SIZE];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  	if(i<num_vec)
    	sdata[tid] = g_idata[i];
 		else
      sdata[tid]=0;
    __syncthreads();
    // do reduction in shared mem
    for(unsigned int s=blockDim.x/2; s > 0; s >>= 1) {
        if(tid < s){
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


void sumVecSec(int *A,int N ,int *r){
  int value=0;
	for(int i=0;i<N;i++)
		value+=A[i];
  *r=value;
  
}

void inicializar(int *A,int N,int a){
	for(int  i = 0; i <N; i++ )
        A[i] = a;
}




int main(){
  int N=3000000;
  int secuencial;
  int mem=(N)*sizeof(int);
  int *A=(int*)malloc(mem);
  int *C=(int*)malloc(mem);
  
  inicializar(A,N,1);
  inicializar(C,N,0);
 
  /*-------------------------------------Algoritmo secuencial-------------------------------------*/
  clock_t start = clock();      
  sumVecSec(A,N,&secuencial);
  clock_t end= clock(); 
  double elapsed_seconds=end-start;
  printf("Tiempo transcurrido Secuencial: %lf\n", (elapsed_seconds / CLOCKS_PER_SEC));
  /*----------------------------------------------------------------------------------------------*/
  
  
  /*-------------------------------------Algoritmo paralelo---------------------------------------*/
  int *d_A=(int*)malloc(mem);
  int *d_C=(int*)malloc(mem);
  cudaMalloc((void**)&d_A,mem);
  cudaMalloc((void**)&d_C,mem);
  
  clock_t start2 = clock(); 
  cudaMemcpy(d_A, A, mem, cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, C, mem, cudaMemcpyHostToDevice);
  float blocksize=BLOCK_SIZE;
   
  int i=N;
  while(i>1){
  	dim3 dimBlock(BLOCK_SIZE,1,1);
    int grid=ceil(i/blocksize);
    dim3 dimGrid(grid,1,1);
    
    sumVecReduce<<<dimGrid,dimBlock>>>(d_A,d_C,i);
		cudaDeviceSynchronize();
  	cudaMemcpy(d_A, d_C, mem, cudaMemcpyDeviceToDevice);
    i=ceil(i/blocksize);
  }
  cudaMemcpy(C, d_C, mem, cudaMemcpyDeviceToHost);
 
  clock_t end2= clock(); 
  double elapsed_seconds2=end2-start2;
  printf("Tiempo transcurrido Paralelo Reduccion: %lf\n", (elapsed_seconds2 / CLOCKS_PER_SEC));  
	
  /*----------------------------------------------------------------------------------------------*/
  
  if(secuencial==C[0])
    printf("Las sumatorias son iguales: %d %d \n",secuencial,C[0]);
  else
  	printf("Las sumatorias no son iguales: %d %d \n",secuencial,C[0]);
	
  printf("Aceleración: %lf",elapsed_seconds/elapsed_seconds2);
     
  free(A);
  free(C);
  cudaFree(d_A);
  cudaFree(d_C);
  return 0;
}
