#include <stdio.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <iostream>
#include <highgui.h>
#include <cv.h>

#define V 200
#define E 200
#define MAX_WEIGHT 1000000
#define TRUE    1
#define FALSE   0
#define USECPSEC 1000000ULL


unsigned long long dtime_usec(unsigned long long prev){
  timeval tv1;
  gettimeofday(&tv1,0);
  return ((tv1.tv_sec * USECPSEC)+tv1.tv_usec) - prev;
  // return ((tv1.tv_sec *1000)+tv1.tv_usec/1000) - prev;
}

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

const Edge edd[E] = {{0, 4}, {0, 6}, {0,2}, {4,6}, {4,7}, {0, 7}, {7, 3}, {3, 1}, {2,5}, {2, 1}, {5,3}, {2,6}, {6,9}, {5,12}, {8,10}, {12,15}, {10,12}, {7, 9}, {7, 10}, {8, 10}, {7,8 },{8,9 },{9,10 },{10,11 },{11,12 },{12,13 },{13,14 },{14,15 },{15,16 },{16,17 },{17,18 },{18,19},{19,20 },{20,21},{21,22 },{22, 23}, {23,24}, {24,25}, {25,26}, {26,27}, {27,28}, {28,29}, {29,30}, {30,31}, {31,32}, {32,33}, {33,34}, {34,35}, {35,36}, {36,37}, {37,38}, {38,39}, {39,40}, {40,41}, {41,42}, {42,43}, {43,44}, {44,45}, {45,46}, {46,47}, {47,48}, {48,49}, {49,50}, {50,51}, {51,52}, {52,53}, {53,54}, {54,55}, {55,56}, {56,57}, {57,58}, {58,59}, {59,60}, {60,61}, {61,62}, {62,63}, {63,64}, {64,65}, {65,66}, {66,67}, {67,68}, {68,69}, {69,70}, {70,71}, {71,72}, {72,73}, {73,74}, {74,75}, {75,76}, {76,77}, {77,78}, {78,79}, {79,80}, {80,81}, {81,82}, {82,83}, {83,84}, {84,85}, {85,86}, {86,87}, {87,88}, {88,89}, {89,90}, {90,91}, {91,92}, {92,93}, {93,94}, {94,95}, {95,96}, {96,97}, {97,98}, {98,99}, {99,100}, {100,101}, {101,102}, {102,103}, {103,104}, {104,105}, {105,106}, {106,107}, {107,108}, {108,109}, {109,110}, {110,111}, {111,112}, {112,113}, {113,114}, {114,115}, {115,116}, {116,117}, {117,118}, {118,119}, {119,120}, {120,121}, {121,122}, {122,123}, {123,124}, {124,125}, {125,126}, {126,127}, {127,128}, {128,129}, {129,130}, {130,131}, {131,132}, {132,133}, {133,134}, {134,135}, {135,136}, {136,137}, {137,138}, {138,139}, {139,140}, {140,141}, {141,142}, {142,143}, {143,144}, {144,145}, {145,146}, {146,147}, {147,148}, {148,149}, {149,150}, {150,151}, {151,152}, {152,153}, {153,154}, {154,155}, {155,156}, {156,157}, {157,158}, {158,159}, {159,160}, {160,161}, {161,162}, {162,163}, {163,164}, {164,165}, {165,166}, {166,167}, {167,168}, {168,169}, {169,170}, {170,171}, {171,172}, {172,173}, {173,174}, {174,175}, {175,176}, {176,177}, {177,178}, {178,179}, {179,180}, {180,181}, {181,182}, {182,183}, {183,184}, {184,185}, {185,186}, {186,187}};
const int ww[E] = {10, 90, 30, 20, 20, 50, 10, 20, 10, 10, 10, 70, 10, 90, 30, 20, 20, 50, 10, 20, 10, 10, 10, 20, 10, 90, 30, 20, 20, 50, 10, 20, 10, 10, 10, 20, 25, 48, 20, 50, 35, 10, 54, 52, 53, 34, 49, 12, 38, 22, 42, 19, 34, 47, 33, 46, 33, 41, 13, 42, 51, 21, 56, 11, 28, 38, 14, 27, 27, 23, 42, 53, 15, 40, 37, 56, 18, 38, 13, 43, 43, 17, 36, 26, 34, 51, 13, 48, 23, 49, 34, 20, 45, 48, 14, 19, 17, 15, 24, 52, 41, 49, 45, 35, 48, 31, 21, 49, 21, 25, 29, 26, 48, 37, 20, 59, 60, 13, 50, 27, 23, 36, 35, 41, 56, 36, 44, 11, 23, 19, 32, 55, 11, 48, 44, 27, 53, 27, 27, 32, 23, 46, 44, 31, 24, 57, 10, 55, 18, 55, 29, 14, 10, 22, 16, 36, 29, 32, 18, 20, 47, 10, 35, 38, 34, 21, 16, 56, 58, 45, 44, 27, 56, 12, 24, 26, 45, 60, 35, 56, 34, 49, 12, 23, 46, 59, 34, 49, 45, 13, 11, 49, 60, 38, 32, 47, 51, 48, 32, 26};

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

void Find_VertexCPU(Vertex *vertices, int *length, int *updateLength)
{
    for (int u = 0; u < V; ++u)
    {
        if(vertices[u].visited == FALSE)
        {
            vertices[u].visited = TRUE;
            int v;
            for(v = 0; v < V; v++)
            {   
                int weight = findEdge(vertices[u], vertices[v]);
                if(weight < MAX_WEIGHT)
                {   
                    if(updateLength[v] > length[u] + weight)
                    {
                        updateLength[v] = length[u] + weight;
                    }
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

void Update_PathsCPU(Vertex *vertices, int *length, int *updateLength)
{
    for (int u = 0; u < V; ++u)
    {
        if(length[u] > updateLength[u])
        {

            length[u] = updateLength[u];
            vertices[u].visited = FALSE;
        }

        updateLength[u] = length[u];
        /* code */
    }


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
    Edge ed[E] = {{0, 4}, {0, 6}, {0,2}, {4,6}, {4,7}, {0, 7}, {7, 3}, {3, 1}, {2,5}, {2, 1}, {5,3}, {2,6}, {6,9}, {5,12}, {8,10}, {12,15}, {10,12}, {7, 9}, {7, 10}, {8, 10}, {7,8 },{8,9 },{9,10 },{10,11 },{11,12 },{12,13 },{13,14 },{14,15 },{15,16 },{16,17 },{17,18 },{18,19},{19,20 },{20,21},{21,22 },{22, 23}, {23,24}, {24,25}, {25,26}, {26,27}, {27,28}, {28,29}, {29,30}, {30,31}, {31,32}, {32,33}, {33,34}, {34,35}, {35,36}, {36,37}, {37,38}, {38,39}, {39,40}, {40,41}, {41,42}, {42,43}, {43,44}, {44,45}, {45,46}, {46,47}, {47,48}, {48,49}, {49,50}, {50,51}, {51,52}, {52,53}, {53,54}, {54,55}, {55,56}, {56,57}, {57,58}, {58,59}, {59,60}, {60,61}, {61,62}, {62,63}, {63,64}, {64,65}, {65,66}, {66,67}, {67,68}, {68,69}, {69,70}, {70,71}, {71,72}, {72,73}, {73,74}, {74,75}, {75,76}, {76,77}, {77,78}, {78,79}, {79,80}, {80,81}, {81,82}, {82,83}, {83,84}, {84,85}, {85,86}, {86,87}, {87,88}, {88,89}, {89,90}, {90,91}, {91,92}, {92,93}, {93,94}, {94,95}, {95,96}, {96,97}, {97,98}, {98,99}, {99,100}, {100,101}, {101,102}, {102,103}, {103,104}, {104,105}, {105,106}, {106,107}, {107,108}, {108,109}, {109,110}, {110,111}, {111,112}, {112,113}, {113,114}, {114,115}, {115,116}, {116,117}, {117,118}, {118,119}, {119,120}, {120,121}, {121,122}, {122,123}, {123,124}, {124,125}, {125,126}, {126,127}, {127,128}, {128,129}, {129,130}, {130,131}, {131,132}, {132,133}, {133,134}, {134,135}, {135,136}, {136,137}, {137,138}, {138,139}, {139,140}, {140,141}, {141,142}, {142,143}, {143,144}, {144,145}, {145,146}, {146,147}, {147,148}, {148,149}, {149,150}, {150,151}, {151,152}, {152,153}, {153,154}, {154,155}, {155,156}, {156,157}, {157,158}, {158,159}, {159,160}, {160,161}, {161,162}, {162,163}, {163,164}, {164,165}, {165,166}, {166,167}, {167,168}, {168,169}, {169,170}, {170,171}, {171,172}, {172,173}, {173,174}, {174,175}, {175,176}, {176,177}, {177,178}, {178,179}, {179,180}, {180,181}, {181,182}, {182,183}, {183,184}, {184,185}, {185,186}, {186,187}};
    int w[E] = {10, 90, 30, 20, 20, 50, 10, 20, 10, 10, 10, 70, 10, 90, 30, 20, 20, 50, 10, 20, 10, 10, 10, 20, 10, 90, 30, 20, 20, 50, 10, 20, 10, 10, 10, 20, 25, 48, 20, 50, 35, 10, 54, 52, 53, 34, 49, 12, 38, 22, 42, 19, 34, 47, 33, 46, 33, 41, 13, 42, 51, 21, 56, 11, 28, 38, 14, 27, 27, 23, 42, 53, 15, 40, 37, 56, 18, 38, 13, 43, 43, 17, 36, 26, 34, 51, 13, 48, 23, 49, 34, 20, 45, 48, 14, 19, 17, 15, 24, 52, 41, 49, 45, 35, 48, 31, 21, 49, 21, 25, 29, 26, 48, 37, 20, 59, 60, 13, 50, 27, 23, 36, 35, 41, 56, 36, 44, 11, 23, 19, 32, 55, 11, 48, 44, 27, 53, 27, 27, 32, 23, 46, 44, 31, 24, 57, 10, 55, 18, 55, 29, 14, 10, 22, 16, 36, 29, 32, 18, 20, 47, 10, 35, 38, 34, 21, 16, 56, 58, 45, 44, 27, 56, 12, 24, 26, 45, 60, 35, 56, 34, 49, 12, 23, 46, 59, 34, 49, 45, 13, 11, 49, 60, 38, 32, 47, 51, 48, 32, 26};

  
    int i = 0;
    for(i = 0; i < V; i++)
    {
        Vertex a = { .title =i , .visited=FALSE};
        vertices[i] = a;


    }



    //Allocate space on the device
    // cudaMalloc((void**)&d_V, sizeV);
    // cudaMalloc((void**)&d_E, sizeE);
    // cudaMalloc((void**)&d_L, size);
    // cudaMalloc((void**)&d_C, size);

    //Initial Node
    Vertex root = {0, FALSE};


    //--------------------------------------Dijkstra's Algorithm--------------------------------------//
    root.visited = TRUE;
    
    
    len[root.title] = 0;
    updateLength[root.title] = 0;

    //Copy variables to the Device
    // cudaMemcpy(d_V, vertices, sizeV, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_L, len, size, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_C, updateLength, size, cudaMemcpyHostToDevice);

    // cudaMemcpyToSymbol(M,w,sizeM);
    // cudaMemcpyToSymbol(M2,ed,sizeE);
    

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
    // cudaEventRecord(timeStart, 0);
    double start, end;
    unsigned long long cpu_time = dtime_usec(0);
        
    //Recopy the variables  
    cudaMemcpy(d_L, len, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, updateLength, size, cudaMemcpyHostToDevice);
                    
    //Parallelization
    for(i = 0; i < V; i++){

            //Find_Vertex<<<1, V>>>(d_V, d_E, d_W, d_L, d_C);
        // Find_Vertex<<<1, V>>>(d_V, d_L, d_C);
        Find_VertexCPU(vertices, len, updateLength);

        // Update_Paths<<<1,V>>>(d_V, d_L, d_C);
        Update_PathsCPU(vertices, len, updateLength);
            
    }   
    
    //Timing Events
    // cudaEventRecord(timeEnd, 0);
    // cudaEventSynchronize(timeEnd);
    // cudaEventElapsedTime(&runningTime, timeStart, timeEnd);

    //Copies the results back
    cpu_time = dtime_usec(cpu_time);
    printf("Finished 1. Basic.  Results match. cpu time: %lld\n", cpu_time);
    // cudaMemcpy(len, d_L, size, cudaMemcpyDeviceToHost);

    printArray(len);

    //Running Time
    printf("Running Time: %f ms\n", runningTime);

    //--------------------------------------Dijkstra's Algorithm--------------------------------------//

    //Free up the space
    free(vertices);
    free(len);
    free(updateLength);
    // cudaFree(d_V);
    // cudaFree(d_L);
    // cudaFree(d_C);
    // cudaEventDestroy(timeStart);
    // cudaEventDestroy(timeEnd);

}
