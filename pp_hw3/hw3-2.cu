#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define Block_Size 32
#define LBS 5 // Log Block_Size

const int INF = 1073741823;
void input(char *inFileName);
void output(char *outFileName);
void block_FW(int BS);
__global__ void phase1(int BS, int block_id, int *Dist_GPU, int vertex_num, size_t pitch);
__global__ void phase2(int BS, int block_id, int *Dist_GPU, int vertex_num, size_t pitch);
__global__ void phase3(int BS, int block_id, int *Dist_GPU, int vertex_num, size_t pitch);

int vertex_num, edge_num, vertex_num_origin;
int *Dist, *Dist_GPU;
size_t pitch;

int main(int argc, char* argv[]){
    input(argv[1]);
    block_FW(Block_Size);
    output(argv[2]);
    return 0;
}

void input(char *inFileName){
    // Read vertex num and edge num
    FILE *file = fopen(inFileName, "rb");
    fread(&vertex_num, sizeof(int), 1, file);
    fread(&edge_num, sizeof(int), 1, file);

    vertex_num_origin = vertex_num;
    // Padding
    vertex_num += (Block_Size - vertex_num % Block_Size);
    
    // Allocate memory for Dist and pinned the host memory to accerlate cudaMemcpy
    cudaMallocHost(&Dist, vertex_num*vertex_num*sizeof(int));

    // Initialize Dist
    for(int i = 0; i < vertex_num; i++){
        for(int j = 0; j < vertex_num; j++){
            Dist[i*vertex_num+j] = (i==j&&i<vertex_num_origin)?0:INF;
        }
    }

    // Read edges
    int pair[3];
    for(int i = 0; i < edge_num; i++){
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0]*vertex_num+pair[1]] = pair[2];
    }
    fclose(file);
}

void output(char *outFileName){
    FILE *file = fopen(outFileName, "w");
    for(int i = 0; i < vertex_num_origin; i++){
        fwrite(&Dist[i*vertex_num], sizeof(int), vertex_num_origin, file);
    }
    fclose(file);
    cudaFreeHost(Dist);
}

void block_FW(int BS){
    cudaMallocPitch(&Dist_GPU, &pitch, vertex_num * sizeof(int), vertex_num);
    cudaMemcpy2D(Dist_GPU, pitch, Dist, vertex_num * sizeof(int), vertex_num * sizeof(int), vertex_num, cudaMemcpyHostToDevice);
    int round = vertex_num/Block_Size;
    for(int block_id = 0; block_id < round; block_id++){
        phase1 <<<1, dim3(Block_Size, Block_Size)>>> (Block_Size, block_id, Dist_GPU, vertex_num, pitch/ sizeof(int));
        phase2 <<<dim3(2, round-1), dim3(Block_Size, Block_Size)>>> (Block_Size, block_id, Dist_GPU, vertex_num, pitch/ sizeof(int));
        phase3 <<<dim3(round-1, round-1), dim3(Block_Size, Block_Size)>>> (Block_Size, block_id, Dist_GPU, vertex_num, pitch/ sizeof(int));
    }
    cudaMemcpy2D(Dist, vertex_num * sizeof(int), Dist_GPU, pitch, vertex_num * sizeof(int), vertex_num, cudaMemcpyDeviceToHost);
    cudaFree(Dist_GPU);
}

__global__ void phase1(int BS, int block_id, int *Dist_GPU, int vertex_num, size_t pitch){
    // Get index of thread
    int b_i = block_id<<LBS;
    int b_j = block_id<<LBS;
    int i = threadIdx.y;
    int j = threadIdx.x;

    // Copy data from global memory to shared memory
    __shared__ int s[Block_Size][Block_Size];
    s[i][j] = Dist_GPU[(b_i+i)*pitch+(b_j+j)];

    // Compute phase 1 - dependent phase
    #pragma unroll
    for(int k = 0; k < Block_Size; k++){
        __syncthreads();
        s[i][j] = min(s[i][j], s[i][k]+s[k][j]);
    }

    // Load data from shared memory to global memory
    Dist_GPU[(b_i+i)*pitch+(b_j+j)] = s[i][j];
}

__global__ void phase2(int BS, int block_id, int *Dist_GPU, int vertex_num, size_t pitch){
    // Get index of thread
    // ROW: (blockIdx.x = 1), COL: (blockIdx.y = 0)
    int b_i = (blockIdx.x*block_id+(!blockIdx.x)*(blockIdx.y+(blockIdx.y>=block_id)))<<LBS;
    int b_j = (blockIdx.x*(blockIdx.y+(blockIdx.y>=block_id))+(!blockIdx.x)*block_id)<<LBS;
    int b_k = block_id<<LBS;
    int i = threadIdx.y, j = threadIdx.x;

    __shared__ int s1[Block_Size][Block_Size], s2[Block_Size][Block_Size];
    int new_dist = Dist_GPU[(b_i+i)*pitch+(b_j+j)];
    s1[i][j] = Dist_GPU[(b_i+i)*pitch+(b_k+j)];
    s2[i][j] = Dist_GPU[(b_k+i)*pitch+(b_j+j)];

    __syncthreads();

    // Compute phase 2 - partial dependent phase
    #pragma unroll
    for(int k = 0; k < Block_Size; k++){
        new_dist = min(new_dist, s1[i][k]+s2[k][j]);
    }

    // Load data from shared memory to global memory
    Dist_GPU[(b_i+i)*pitch+(b_j+j)] = new_dist;
}

__global__ void phase3(int BS, int block_id, int *Dist_GPU, int vertex_num, size_t pitch){
    // Get index of thread
    // ROW: (blockIdx.x = 1), COL: (blockIdx.y = 0)
    int b_i = (blockIdx.x+(blockIdx.x>=block_id))<<LBS;
    int b_j = (blockIdx.y+(blockIdx.y>=block_id))<<LBS;
    int b_k = block_id<<LBS;
    int i = threadIdx.y;
    int j = threadIdx.x;

    __shared__ int s1[Block_Size][Block_Size], s2[Block_Size][Block_Size];
    int new_dist = Dist_GPU[(b_i+i)*pitch+(b_j+j)];
    s1[i][j] = Dist_GPU[(b_i+i)*pitch+(b_k+j)];
    s2[i][j] = Dist_GPU[(b_k+i)*pitch+(b_j+j)];
    
    __syncthreads();
    
    // Compute phase 3 - independent phase
    #pragma unroll
    for(int k = 0; k < Block_Size; k++){
        new_dist = min(new_dist, s1[i][k]+s2[k][j]);
    }

    // Load data from shared memory to global memory
    Dist_GPU[(b_i+i)*pitch+(b_j+j)] = new_dist;
}