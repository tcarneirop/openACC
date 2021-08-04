#include <cuda.h>
#include <stdio.h>
#include <string.h>

#define _QUEENS_BLOCK_SIZE_ 	128
#define _VAZIO_      -1


typedef struct queen_root{
	
	unsigned int flag;
	char board[12];
} QueenRoot;


extern "C" int GPU_device_count(){
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    return deviceCount;
}


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

bool MCstillLegal(const char *board, const int r){
  register int i;
  register int ld;
  register int rd;
  // Check vertical
  for ( i = 0; i < r; ++i)
    if (board[i] == board[r]) return false;
    // Check diagonals
    ld = board[r];  //left diagonal columns
    rd = board[r];  // right diagonal columns
    for ( i = r-1; i >= 0; --i) {
      --ld; ++rd;
      if (board[i] == ld || board[i] == rd) return false;
    }

    return true;
}

void prefixesHandleSol(QueenRoot *root_prefixes,unsigned int flag,char *board,int initialDepth,int num_sol){

    root_prefixes[num_sol].flag = flag;

    for(int i = 0; i<initialDepth;++i)
      root_prefixes[num_sol].board[i] = (char)board[i];
}



unsigned int BP_queens_prefixes(int size, int initialDepth ,unsigned long long *tree_size, QueenRoot *root_prefixes){

    register unsigned int flag = 0;
    register int bit_test = 0;
    register char vertice[20]; //representa o ciclo
    register int i, nivel; //para dizer que 0-1 ja foi visitado e a busca comeca de 1, bote 2
    register unsigned long long int local_tree = 0ULL;
    unsigned int num_sol = 0;
   //register int custo = 0;

    /*Inicializacao*/
    for (i = 0; i < size; ++i) { //
        vertice[i] = -1;
    }

    nivel = 0;

    do{

        vertice[nivel]++;
        bit_test = 0;
        bit_test |= (1<<vertice[nivel]);


        if(vertice[nivel] == size){
            vertice[nivel] = _VAZIO_;
                //if(block_ub > upper)   block_ub = upper;
        }else if ( MCstillLegal(vertice, nivel) && !(flag &  bit_test ) ){ //is legal

                flag |= (1ULL<<vertice[nivel]);
                nivel++;
                ++local_tree;
                if (nivel == initialDepth){ //handle solution
                   prefixesHandleSol(root_prefixes,flag,vertice,initialDepth,num_sol);
                   num_sol++;
            }else continue;
        }else continue;

        nivel--;
        flag &= ~(1ULL<<vertice[nivel]);

    }while(nivel >= 0);

    *tree_size = local_tree;

    return num_sol;
}



__device__  bool GPU_queens_stillLegal(const char *board, const int r){

  bool safe = true;
  int i;
  register int ld;
  register int rd;
  // Check vertical
  for ( i = 0; i < r; ++i)
    if (board[i] == board[r]) safe = false;
    // Check diagonals
    ld = board[r];  //left diagonal columns
    rd = board[r];  // right diagonal columns
    for ( i = r-1; i >= 0; --i) {
      --ld; ++rd;
      if (board[i] == ld || board[i] == rd) safe = false;
    }

    return safe;
}


__global__ void BP_queens_root_dfs(int N, unsigned int nPreFixos, int depthPreFixos,
    QueenRoot *root_prefixes,unsigned int *vector_of_tree_size, unsigned int *sols){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nPreFixos) {
        register unsigned int flag = 0;
        register unsigned int bit_test = 0;
        register char vertice[20]; //representa o ciclo
        register int N_l = N;
        register int i, depth; 
        register int qtd_solucoes_thread = 0;
        register int depthGlobal = depthPreFixos;
        register unsigned int tree_size = 0;

        #pragma unroll 2
        for (i = 0; i < N_l; ++i) {
            vertice[i] = _VAZIO_;
        }

        flag = root_prefixes[idx].flag;

        #pragma unroll 2
        for (i = 0; i < depthGlobal; ++i)
            vertice[i] = root_prefixes[idx].board[i];

        depth=depthGlobal;

        do{

            vertice[depth]++;
            bit_test = 0;
            bit_test |= (1<<vertice[depth]);

            if(vertice[depth] == N_l){
                vertice[depth] = _VAZIO_;
                //if(block_ub > upper)   block_ub = upper;
            }else if (!(flag &  bit_test ) && GPU_queens_stillLegal(vertice, depth)){

                    ++tree_size;
                    flag |= (1ULL<<vertice[depth]);

                    depth++;

                    if (depth == N_l) { //sol
                        ++qtd_solucoes_thread; 
                    }else continue;
                }else continue;

            depth--;
            flag &= ~(1ULL<<vertice[depth]);

            }while(depth >= depthGlobal); //FIM DO DFS_BNB

        sols[idx] = qtd_solucoes_thread;
        vector_of_tree_size[idx] = tree_size;
    }//if
}//kernel
////////


void GPU_call_cuda_queens(short size, int initial_depth, unsigned int n_explorers, QueenRoot *root_prefixes_h ,
    
	unsigned int *vector_of_tree_size_h, unsigned int *sols_h, int gpu_id){
    cudaSetDevice(gpu_id);
    //cudaFuncSetCacheConfig(BP_queens_root_dfs,cudaFuncCachePreferL1);
   

    unsigned int *vector_of_tree_size_d;
    unsigned int *sols_d;
    QueenRoot *root_prefixes_d;

    int num_blocks = ceil((double)n_explorers/_QUEENS_BLOCK_SIZE_);

    cudaMalloc((void**) &vector_of_tree_size_d,n_explorers*sizeof(unsigned int));
    cudaMalloc((void**) &sols_d,n_explorers*sizeof(unsigned int));
    cudaMalloc((void**) &root_prefixes_d,n_explorers*sizeof(QueenRoot));

    //I Think this is not possible in Chapel. It must be internal
    cudaMemcpy(root_prefixes_d, root_prefixes_h, n_explorers * sizeof(QueenRoot), cudaMemcpyHostToDevice);

    printf("\n### Regular BP-DFS search. ###\n");
    
    //kernel_start =  rtclock();
    
    BP_queens_root_dfs<<< num_blocks,_QUEENS_BLOCK_SIZE_>>> (size,n_explorers,initial_depth,root_prefixes_d, vector_of_tree_size_d,sols_d);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //kernel_stop = rtclock();

    cudaMemcpy(vector_of_tree_size_h,vector_of_tree_size_d,n_explorers*sizeof(unsigned int),cudaMemcpyDeviceToHost);
    cudaMemcpy(sols_h,sols_d,n_explorers*sizeof(unsigned int),cudaMemcpyDeviceToHost);

    
    cudaFree(vector_of_tree_size_d);
    cudaFree(sols_d);
    cudaFree(root_prefixes_d);
    //After that, Chapel reduces the values
}


unsigned long long get_tree_size(unsigned *subtree_values, unsigned survivors){
    
    unsigned long long acumulator = 0ULL;

    // #pragma omp parallel for schedule(static) reduction(+:acumulator) 
    for(unsigned int i = 0; i<survivors; ++i){
        acumulator+=subtree_values[i];
     }

     return acumulator;
}

int main(){


	short size = 15;
	int initial_depth = 7;

	unsigned  max_number_prefixes = 75580635;
	
	unsigned  *vector_of_tree_size_h = (unsigned*)malloc(sizeof(unsigned) * max_number_prefixes);
	unsigned  *sols_h = (unsigned*)malloc(sizeof(unsigned) * max_number_prefixes);
	QueenRoot *active_set_h = (QueenRoot*)malloc(sizeof(QueenRoot)* max_number_prefixes);
	
	unsigned long long tree_size = 0ULL;
    unsigned long long qtd_sols_global = 0ULL;
  	unsigned long long  initial_tree_size = 0ULL;
  	unsigned long long gpu_tree_size = 0ULL;

	int gpu_id = 0; //For them multigpu code

	unsigned int n_explorers = BP_queens_prefixes(size,initial_depth ,&initial_tree_size,active_set_h);

	
	printf("\nProblem size: %d \nInitial depth: %d\n", size, initial_depth );

	GPU_call_cuda_queens(size, initial_depth, n_explorers, active_set_h , vector_of_tree_size_h, sols_h, gpu_id);
  
    qtd_sols_global = get_tree_size(sols_h,n_explorers); //i'm using get_tree_size to make a reduction... not beautiful

    tree_size = get_tree_size(vector_of_tree_size_h,n_explorers);

    tree_size+=initial_tree_size;
 	gpu_tree_size = tree_size-initial_tree_size;

	printf("\nNumber of sol: %llu\nFinal Tree size: %llu \n\tInitial Tree size: %llu \n\tGPU Tree size: %llu\n\t", qtd_sols_global, tree_size, initial_tree_size, gpu_tree_size );
	
	

	return 0;

}
