#include <stdio.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define V 24
#define E 36
#define MAX_WEIGHT 1000000
#define TRUE    1
#define FALSE   0


typedef int boolean;
//
//Represents an edge or path between Vertices
typedef struct
{
	int u;
	int v;

} Edge;

//Represents a Vertex
typedef struct 
{
	int title;
	boolean visited;	

} Vertex;

__constant__ int M[E];
__constant__ Edge M2[E];

const Edge edd[E] = {{0, 4}, {0, 6}, {0,2}, {4,6}, {4,7}, {0, 7}, {7, 3}, {3, 1}, {2,5}, {2, 1}, {5,3}, {2,6}, {6,9}, {5,12}, {8,10}, {12,15}, {10,12}, {7, 9}, {7, 10}, {8, 10}, {7,8 },{8,9 },{9,10 },{10,11 },{11,12 },{12,13 },{13,14 },{14,15 },{15,16 },{16,17 },{17,18 },{18,19},{19,20 },{20,21},{21,22 },{22, 23}};
const int ww[E] = {10, 90, 30, 20, 20, 50, 10, 20, 10, 10, 10, 70, 10, 90, 30, 20, 20, 50, 10, 20, 10, 10, 10, 20, 10, 90, 30, 20, 20, 50, 10, 20, 10, 10, 10, 20};

//Finds the weight of the path from vertex u to vertex v
__device__ int findEdgeD(Vertex u, Vertex v)
{
	
	int i;
	for(i = 0; i < E; i++)
	{

		if(M2[i].u == u.title && M2[i].v == v.title)
		{
			return M[i];
		}
	}

	return MAX_WEIGHT;

}

int findEdge(Vertex u, Vertex v)
{
	
	int i;
	for(i = 0; i < E; i++)
	{

		if(edd[i].u == u.title && edd[i].v == v.title)
		{
			return ww[i];
		}
	}

	return MAX_WEIGHT;

}

//Finds the branches of the vertex
__global__ void Find_Vertex(Vertex *vertices, int *length, int *updateLength)
{

	int u = threadIdx.x;


	if(vertices[u].visited == FALSE)
	{
		

		vertices[u].visited = TRUE;


		int v;

		for(v = 0; v < V; v++)
		{	
			//Find the weight of the edge
			int weight = findEdgeD(vertices[u], vertices[v]);

			//Checks if the weight is a candidate
			if(weight < MAX_WEIGHT)
			{	
				//If the weight is shorter than the current weight, replace it
				if(updateLength[v] > length[u] + weight)
				{
					updateLength[v] = length[u] + weight;
				}
			}
		}

	}
	
}

//Updates the shortest path array (length)
__global__ void Update_Paths(Vertex *vertices, int *length, int *updateLength)
{
	int u = threadIdx.x;
	if(length[u] > updateLength[u])
	{

		length[u] = updateLength[u];
		vertices[u].visited = FALSE;
	}

	updateLength[u] = length[u];


}


//Prints the an array of elements
void printArray(int *array)
{
	int i;
	for(i = 0; i < V; i++)
	{
		printf("Shortest Path to Vertex: %d is %d\n", i, array[i]);
	}
}


//Runs the program
int main(void)
{

	//Variables for the Host Device
	Vertex *vertices;

	//Len is the shortest path and updateLength is a special array for modifying updates to the shortest path
	int *len, *updateLength;
	


	//Pointers for the CUDA device
	Vertex *d_V;
	Edge *d_E;
	int *d_L;
	int *d_C;
  
  int sizeM = sizeof(int)*E; 

	//Sizes used for allocation
	int sizeV = sizeof(Vertex) * V;
	int sizeE = sizeof(Edge) * E;
	int size = V * sizeof(int);


	//Timer initialization
	float runningTime;
	cudaEvent_t timeStart, timeEnd;


	//Creates the timers
	cudaEventCreate(&timeStart);
	cudaEventCreate(&timeEnd);


	//Allocates space for the variables
	vertices = (Vertex *)malloc(sizeV);
	len = (int *)malloc(size);
	updateLength = (int *)malloc(size);

	
	//----------------------------------Graph Base Test-------------------------------------//
	Edge ed[E] = {{0, 4}, {0, 6}, {0,2}, {4,6}, {4,7}, {0, 7}, {7, 3}, {3, 1}, {2,5}, {2, 1}, {5,3}, {2,6}, {6,9}, {5,12}, {8,10}, {12,15}, {10,12}, {7, 9}, {7, 10}, {8, 10}, {7,8 },{8,9 },{9,10 },{10,11 },{11,12 },{12,13 },{13,14 },{14,15 },{15,16 },{16,17 },{17,18 },{18,19},{19,20 },{20,21},{21,22 },{22, 23}};
	int w[E] = {10, 90, 30, 20, 20, 50, 10, 20, 10, 10, 10, 70, 10, 90, 30, 20, 20, 50, 10, 20, 10, 10, 10, 20, 10, 90, 30, 20, 20, 50, 10, 20, 10, 10, 10, 20};

  
	int i = 0;
	for(i = 0; i < V; i++)
	{
		Vertex a = { .title =i , .visited=FALSE};
		vertices[i] = a;


	}

	//----------------------------------Graph Base Test-------------------------------------//


	//--------------------------------Graph Randomizer-----------------------------------//
	// srand(time(NULL));
	// int i = 0;
	// for(i = 0; i < V; i++)
	// {
	// 	Vertex a = { .title =(int) i, .visited=FALSE};
	// 	vertices[i] = a;


	// }



	// for(i = 0; i < E; i++)
	// {

	// 	Edge e = {.u = (int) rand()%V , .v = rand()%V};
	// 	edges[i] = e;

	// 	weights[i] = rand()%100;

	// }

	//--------------------------------Graph Randomizer-----------------------------------//


	//Allocate space on the device
	cudaMalloc((void**)&d_V, sizeV);
	cudaMalloc((void**)&d_E, sizeE);
	cudaMalloc((void**)&d_L, size);
	cudaMalloc((void**)&d_C, size);

	//Initial Node
	Vertex root = {0, FALSE};


	//--------------------------------------Dijkstra's Algorithm--------------------------------------//
	root.visited = TRUE;
	
	
	len[root.title] = 0;
	updateLength[root.title] = 0;

	//Copy variables to the Device
	cudaMemcpy(d_V, vertices, sizeV, cudaMemcpyHostToDevice);
	cudaMemcpy(d_L, len, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, updateLength, size, cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol(M,w,sizeM);
  	cudaMemcpyToSymbol(M2,ed,sizeE);
	

	//Loop finds the initial paths from the node 's'
	for(i = 0; i < V;i++)
	{

		if(vertices[i].title != root.title)
		{
			len[(int)vertices[i].title] = findEdge(root, vertices[i]);
			updateLength[vertices[i].title] = len[(int)vertices[i].title];
			
			

		}
		else{
		
			vertices[i].visited = TRUE;
		}


	}

	//Start the timer
	cudaEventRecord(timeStart, 0);
		
	//Recopy the variables	
	cudaMemcpy(d_L, len, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, updateLength, size, cudaMemcpyHostToDevice);
					
	//Parallelization
	for(i = 0; i < V; i++){

	//Find_Vertex<<<1, V>>>(d_V, d_E, d_W, d_L, d_C);
    	Find_Vertex<<<1, V>>>(d_V, d_L, d_C);

	Update_Paths<<<1,V>>>(d_V, d_L, d_C);
			
	}	
	
	//Timing Events
	cudaEventRecord(timeEnd, 0);
	cudaEventSynchronize(timeEnd);
	cudaEventElapsedTime(&runningTime, timeStart, timeEnd);

	//Copies the results back
	cudaMemcpy(len, d_L, size, cudaMemcpyDeviceToHost);

	printArray(len);

	//Running Time
	printf("Running Time: %f ms\n", runningTime);

	//--------------------------------------Dijkstra's Algorithm--------------------------------------//

	//Free up the space
	free(vertices);
	free(len);
	free(updateLength);
	cudaFree(d_V);
	cudaFree(d_L);
	cudaFree(d_C);
	cudaEventDestroy(timeStart);
	cudaEventDestroy(timeEnd);

}
