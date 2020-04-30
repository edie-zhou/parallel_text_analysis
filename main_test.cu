#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <iostream>
#include "cuda_runtime.h"
#include <iomanip>      // std::setprecision

#include "device_launch_parameters.h"

#include "input.h"

#define NUM_THREADS_PER_BLOCK 512

int* create_shifts (char* pattern);


__global__ void horspool_match (char* text, char* pattern, int* shift_table, unsigned int* num_matches, int chunk_size,
    int num_chunks, int text_size, int pat_len, unsigned int* d_output);
__global__ void prescan(int *g_odata, int *g_idata, int n);

int linear_horspool_match (char* text, char* pattern, int* shift_table, unsigned int* num_matches, int chunk_size,
        int num_chunks, int text_size, int pat_len, int myId) ;

void sum_scan_blelloch(unsigned int* d_out, unsigned int* d_in, size_t numElems);
__global__ void gpu_add_block_sums(unsigned int* d_out, unsigned int* d_in, unsigned int* d_block_sums,
    size_t numElems);

__global__ void gpu_prescan(unsigned int* d_out, unsigned int* d_in, unsigned int* d_block_sums,
    unsigned int len, unsigned int shmem_sz, unsigned int max_elems_per_block);

using namespace std;

int determineNumBlocks(vector<string_chunk> chunks) {
	int numBlocks = 0;
	for (int i = 0; i < chunks.size(); i = i + NUM_THREADS_PER_BLOCK) {
		numBlocks++;
	}
	return numBlocks;
}

int main(int argc, char* argv[])
{
	Input inputObj;

	char* flatText = inputObj.flattenText();
	char* testPattern = (char*)malloc(5 * sizeof(char));
    testPattern = strcpy(testPattern, "test");
    testPattern[4] = '\0';
    int* skipTable = create_shifts(testPattern);
	unsigned int* numMatches = (unsigned int*)malloc(1 * sizeof(unsigned int));
	*numMatches = 0;

	int fullTextSize = inputObj.getChunks().size() * CHUNK_SIZE * sizeof(char);
	int patternSize = strlen(testPattern) * sizeof(char);
	int skipTableSize = 126 * sizeof(int);

	char* d_fullText;
	char* d_testPattern;
	int* d_skipTable;
	unsigned int* d_numMatches;
    unsigned int* d_output;

	cudaMalloc((void**)& d_fullText, fullTextSize);
	cudaMalloc((void**)& d_testPattern, patternSize);
	cudaMalloc((void**)& d_skipTable, skipTableSize);
	cudaMalloc((void**)& d_numMatches, sizeof(unsigned int));

	cudaMemcpy(d_fullText, flatText, fullTextSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_testPattern, testPattern, patternSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_skipTable, skipTable, skipTableSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_numMatches, numMatches, sizeof(unsigned int), cudaMemcpyHostToDevice);

    int numBlocks = determineNumBlocks(inputObj.getChunks());
    cudaMalloc((void**)& d_output, numBlocks * sizeof(unsigned int));
    unsigned int* output = (unsigned int *)malloc(sizeof(unsigned int) * numBlocks);
    for(int i = 0; i < numBlocks; i++){
        output[i] = 0;
    }
    cudaMemcpy(d_output, output, numBlocks * sizeof(int), cudaMemcpyHostToDevice);

    time_t start, end , start1,end1 = 0;
    int text_len = strlen(flatText);
    int pat_len = strlen(testPattern); 
    int num_chunks = inputObj.getChunks().size(); 
    cudaDeviceSynchronize();

    //time(&start);   
    start = clock();

	horspool_match << <numBlocks, NUM_THREADS_PER_BLOCK, NUM_THREADS_PER_BLOCK * sizeof(int) >> > (d_fullText, d_testPattern, d_skipTable, d_numMatches, CHUNK_SIZE, 
        num_chunks , text_len, pat_len, d_output);
        cudaDeviceSynchronize();
    
    unsigned int* d_result;
    cudaMalloc(&d_result, sizeof(unsigned int) * numBlocks);
    sum_scan_blelloch(d_result, d_output, numBlocks);
    end = clock();
    
    unsigned int* result_arr = (unsigned int*) malloc (numBlocks * sizeof(unsigned int));
    cudaMemcpy(result_arr, d_result, sizeof(unsigned int) * numBlocks, cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    
    for (int k = 0; k < numBlocks; ++k) {
        printf("index: %d, val: %d\n", k, result_arr[k]);
    }
    
    free(result_arr);
    
    start1 = clock();
    unsigned int result = 0;
    for(int myId =0; myId < numBlocks * NUM_THREADS_PER_BLOCK; myId++){
        result += linear_horspool_match(flatText, testPattern, skipTable, numMatches, CHUNK_SIZE, 
            num_chunks , text_len, pat_len, myId);    
    }
    end1 = clock();
    //cout << "result is " << result << endl;
    *numMatches = result;
    cudaDeviceSynchronize();

    // Calculating total time taken by the program. 
    double time_taken = double(end - start)/ CLOCKS_PER_SEC; 
    cout << "Time taken by program is : " << setprecision(9) << time_taken << endl; 
    time_taken = double(end1 - start1)/ CLOCKS_PER_SEC;
    cout << "Time taken by program1 is : " << setprecision(9) << time_taken << endl; 
    cudaMemcpy(numMatches, d_numMatches, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(output, d_output, sizeof(int) * numBlocks, cudaMemcpyDeviceToHost);

    cout << "Number of Matches1: " << result << endl;
    int answer= 0; 
    for(int i = 0; i < numBlocks-1; i++){
        answer += output[i];
    }
    cout << "Number of Matches: " << *numMatches << endl;


	cudaFree(d_fullText); cudaFree(d_testPattern); cudaFree(d_skipTable); cudaFree(d_numMatches);
	

	free(testPattern);
	free(skipTable);
	free(numMatches);
}

int linear_horspool_match (char* text, char* pattern, int* shift_table, unsigned int* num_matches, int chunk_size,
    int num_chunks, int text_size, int pat_len, int myId) {

        int count = 0;
        int text_length = (chunk_size * myId) + chunk_size + pat_len - 1;
    
        // don't need to check first pattern_length - 1 characters
        int i = (myId*chunk_size) + pat_len - 1;
        int k = 0;
        while(i < text_length) {
            // reset matched character count
            k = 0;
    
            if (i >= text_size) {
            // break out if i tries to step past text length
                break;
            }
    
            while(k <= pat_len - 1 && pattern[pat_len - 1 - k] == text[i - k]) {
            // increment matched character count
                k++;
            }
            if(k == pat_len) {
            // increment pattern count, text index
                ++count;
                ++i;
    
            } else {
                i = i + shift_table[text[i]];
            }
        }
        return count;
        // Add count to total matches atomically
    
    }
    

/**
 *  Purpose:
 *    Boyer-Moore-Horspool pattern matching algorithm implementation
 * 
 *  Args:
 *    text        {char*}: Text c-string - still text
 *    pattern     {char*}: Target c-string - still pattern
 *    shift_table  {int*}: Skip table - shift table
 *    num_matches   {int}: Total match count - num_matches
 *    chunk_size    {int}: Length of chunk size
 *    num_chunks    {int}: Total number of chunks
 *    text_size     {int}: Integer text length
 *    pat_len       {int}: Integer pattern length
 *  Returns:
 *    None
 */ 
 

 __global__ void horspool_match (char* text, char* pattern, int* shift_table, unsigned int* num_matches, int chunk_size,
    int num_chunks, int text_size, int pat_len , unsigned int* d_output) {
    extern __shared__ int s[];


    int count = 0;
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    if(myId > num_chunks){ //if thread is an invalid thread
        return;
    }

    int text_length = (chunk_size * myId) + chunk_size + pat_len - 1;

    // don't need to check first pattern_length - 1 characters
    int i = (myId*chunk_size) + pat_len - 1;
    int k = 0;
    while(i < text_length) {
        // reset matched character count
        k = 0;

        if (i >= text_size) {
        // break out if i tries to step past text length
            break;
        }

        while(k <= pat_len - 1 && pattern[pat_len - 1 - k] == text[i - k]) {
        // increment matched character count
            k++;
        }
        if(k == pat_len) {
        // increment pattern count, text index
            ++count;
            ++i;

        } else {
            i = i + shift_table[text[i]];
        }
    }

    // atomicAdd(num_matches, count);
    s[threadIdx.x] = count;
    __syncthreads();

    // Add count to total matches atomically
    if (threadIdx.x == 0 ){
        int sum = 0; 
        for(int idx =0; idx < NUM_THREADS_PER_BLOCK; idx++){
            sum += s[idx];
        }
        d_output[blockIdx.x] = sum;
    }
}


/**
 *  Purpose:
 *    Create shift table for Boyer-Moore-Horspool algorithm
 *  
 *  Args:
 *    pattern {char*}: desired pattern c-string
 */ 
int* create_shifts (char* pattern)
{
    // Offset for first ASCII character

    // Line break ASCII value
    const char LINE_BREAK = '\n';

    // Printable ASCII chars are 32-126 inclusive, line break is 10
    const int TABLE_SIZ = 126;

    int length = strlen(pattern);
    int* shift_table = (int*) malloc (sizeof(int) * TABLE_SIZ);

    for(int i = 0; i < TABLE_SIZ; i++) {
        // set all entries to longest shift (pattern length)
        shift_table[i] = length;
    }
    for(int j = 0; j < length - 1; j++) {
        // set pattern characters to shortest shifts
        shift_table[pattern[j]] = length - 1 - j;
    }

    // assign shift of 1 for line breaks
    shift_table[LINE_BREAK] = 1;

    return shift_table;
}

#define MAX_BLOCK_SZ 1024
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5

#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(n) \
	((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#endif

__global__
void gpu_add_block_sums(unsigned int* d_out,
	unsigned int* d_in,
	unsigned int* d_block_sums,
	size_t numElems)
{
	//unsigned int glbl_t_idx = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int d_block_sum_val = d_block_sums[blockIdx.x];

	// Simple implementation's performance is not significantly (if at all)
	//  better than previous verbose implementation
	unsigned int cpy_idx = 2 * blockIdx.x * blockDim.x + threadIdx.x;
	if (cpy_idx < numElems)
	{
		d_out[cpy_idx] = d_in[cpy_idx] + d_block_sum_val;
		if (cpy_idx + blockDim.x < numElems)
			d_out[cpy_idx + blockDim.x] = d_in[cpy_idx + blockDim.x] + d_block_sum_val;
	}
}

// Modified version of Mark Harris' implementation of the Blelloch scan
//  according to https://www.mimuw.edu.pl/~ps209291/kgkp/slides/scan.pdf
__global__
void gpu_prescan(unsigned int* d_out,
	unsigned int* d_in,
	unsigned int* d_block_sums,
	unsigned int len,
	unsigned int shmem_sz,
	unsigned int max_elems_per_block)
{
	// Allocated on invocation
	extern __shared__ unsigned int s_out[];

	int thid = threadIdx.x;
	int ai = thid;
	int bi = thid + blockDim.x;

	// Zero out the shared memory
	// Helpful especially when input size is not power of two
	s_out[thid] = 0;
	s_out[thid + blockDim.x] = 0;
	// If CONFLICT_FREE_OFFSET is used, shared memory
	//  must be a few more than 2 * blockDim.x
	if (thid + max_elems_per_block < shmem_sz)
		s_out[thid + max_elems_per_block] = 0;

	__syncthreads();
	
	// Copy d_in to shared memory
	// Note that d_in's elements are scattered into shared memory
	//  in light of avoiding bank conflicts
	unsigned int cpy_idx = max_elems_per_block * blockIdx.x + threadIdx.x;
	if (cpy_idx < len)
	{
		s_out[ai + CONFLICT_FREE_OFFSET(ai)] = d_in[cpy_idx];
		if (cpy_idx + blockDim.x < len)
			s_out[bi + CONFLICT_FREE_OFFSET(bi)] = d_in[cpy_idx + blockDim.x];
	}

	// For both upsweep and downsweep:
	// Sequential indices with conflict free padding
	//  Amount of padding = target index / num banks
	//  This "shifts" the target indices by one every multiple
	//   of the num banks
	// offset controls the stride and starting index of 
	//  target elems at every iteration
	// d just controls which threads are active
	// Sweeps are pivoted on the last element of shared memory

	// Upsweep/Reduce step
	int offset = 1;
	for (int d = max_elems_per_block >> 1; d > 0; d >>= 1)
	{
		__syncthreads();

		if (thid < d)
		{
			int ai = offset * ((thid << 1) + 1) - 1;
			int bi = offset * ((thid << 1) + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			s_out[bi] += s_out[ai];
		}
		offset <<= 1;
	}

	// Save the total sum on the global block sums array
	// Then clear the last element on the shared memory
	if (thid == 0) 
	{ 
		d_block_sums[blockIdx.x] = s_out[max_elems_per_block - 1 
			+ CONFLICT_FREE_OFFSET(max_elems_per_block - 1)];
		s_out[max_elems_per_block - 1 
			+ CONFLICT_FREE_OFFSET(max_elems_per_block - 1)] = 0;
	}

	// Downsweep step
	for (int d = 1; d < max_elems_per_block; d <<= 1)
	{
		offset >>= 1;
		__syncthreads();

		if (thid < d)
		{
			int ai = offset * ((thid << 1) + 1) - 1;
			int bi = offset * ((thid << 1) + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			unsigned int temp = s_out[ai];
			s_out[ai] = s_out[bi];
			s_out[bi] += temp;
		}
	}
	__syncthreads();

	// Copy contents of shared memory to global memory
	if (cpy_idx < len)
	{
		d_out[cpy_idx] = s_out[ai + CONFLICT_FREE_OFFSET(ai)];
		if (cpy_idx + blockDim.x < len)
			d_out[cpy_idx + blockDim.x] = s_out[bi + CONFLICT_FREE_OFFSET(bi)];
	}
}

 
void sum_scan_blelloch(unsigned int* d_out, unsigned int* d_in, size_t numElems)
{
	// Zero out d_out
	cudaMemset(d_out, 0, numElems * sizeof(unsigned int));

	// Set up number of threads and blocks
	
	unsigned int block_sz = MAX_BLOCK_SZ / 2;
	unsigned int max_elems_per_block = 2 * block_sz; // due to binary tree nature of algorithm

	// If input size is not power of two, the remainder will still need a whole block
	// Thus, number of blocks must be the ceiling of input size / max elems that a block can handle
	//unsigned int grid_sz = (unsigned int) std::ceil((double) numElems / (double) max_elems_per_block);
	// UPDATE: Instead of using ceiling and risking miscalculation due to precision, just automatically  
	//  add 1 to the grid size when the input size cannot be divided cleanly by the block's capacity
	unsigned int grid_sz = numElems / max_elems_per_block;
	// Take advantage of the fact that integer division drops the decimals
	if (numElems % max_elems_per_block != 0) 
		grid_sz += 1;

	// Conflict free padding requires that shared memory be more than 2 * block_sz
	unsigned int shmem_sz = max_elems_per_block + ((max_elems_per_block - 1) >> LOG_NUM_BANKS);

	// Allocate memory for array of total sums produced by each block
	// Array length must be the same as number of blocks
	unsigned int* d_block_sums;
	cudaMalloc(&d_block_sums, sizeof(unsigned int) * grid_sz);
	cudaMemset(d_block_sums, 0, sizeof(unsigned int) * grid_sz);

	// Sum scan data allocated to each block
	gpu_prescan<<<grid_sz, block_sz, sizeof(unsigned int) * shmem_sz>>>(d_out, 
																	d_in, 
																	d_block_sums, 
																	numElems, 
																	shmem_sz,
																	max_elems_per_block);

	// Sum scan total sums produced by each block
	// Use basic implementation if number of total sums is <= 2 * block_sz
	//  (This requires only one block to do the scan)
	if (grid_sz <= max_elems_per_block)
	{
		unsigned int* d_dummy_blocks_sums;
		cudaMalloc(&d_dummy_blocks_sums, sizeof(unsigned int));
		cudaMemset(d_dummy_blocks_sums, 0, sizeof(unsigned int));
		gpu_prescan<<<1, block_sz, sizeof(unsigned int) * shmem_sz>>>(d_block_sums, 
																	d_block_sums, 
																	d_dummy_blocks_sums, 
																	grid_sz, 
																	shmem_sz,
																	max_elems_per_block);
		cudaFree(d_dummy_blocks_sums);
	}
	// Else, recurse on this same function as you'll need the full-blown scan
	//  for the block sums
	else
	{
		unsigned int* d_in_block_sums;
		cudaMalloc(&d_in_block_sums, sizeof(unsigned int) * grid_sz);
		cudaMemcpy(d_in_block_sums, d_block_sums, sizeof(unsigned int) * grid_sz, cudaMemcpyDeviceToDevice);
		sum_scan_blelloch(d_block_sums, d_in_block_sums, grid_sz);
		cudaFree(d_in_block_sums);
	}

	// Add each block's total sum to its scan output
	// in order to get the final, global scanned array
	gpu_add_block_sums<<<grid_sz, block_sz>>>(d_out, d_out, d_block_sums, numElems);

	cudaFree(d_block_sums);
}

