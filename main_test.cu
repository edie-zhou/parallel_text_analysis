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
int get_line_start (char* text, int idx);
int get_line_end (char* text, int idx, int pattern_len);
void print_line (char* text, int start_index, int end_index, int pat_start, int pat_len);

__global__ void horspool_match (char* text, char* pattern, int* shift_table, unsigned int* num_matches, int chunk_size,
    int* map, int* lineData, int num_chunks, int text_size, int pat_len, int * d_output);

int linear_horspool_match (char* text, char* pattern, int* shift_table, unsigned int* num_matches, int chunk_size,
        int* map, int* lineData, int num_chunks, int text_size, int pat_len, int myId) ;
    


using namespace std;

long * calcIndexes(long num_strings, long length){
    long * indexArray = ( long *)malloc(sizeof( long) * (num_strings+2));

    long i;
    long sum ;
   for (sum=0, i=0; i < num_strings; i++, sum+=CHUNK_SIZE){
       indexArray[i] = sum;
   }
   indexArray[i] = sum + (length - ((num_strings-1) * CHUNK_SIZE));
   indexArray[++i] = NULL;
   return indexArray;
}

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
	int* map = inputObj.getMap();
	int* lineData = inputObj.getLineData();

	int fullTextSize = inputObj.getChunks().size() * CHUNK_SIZE * sizeof(char);
	int patternSize = strlen(testPattern) * sizeof(char);
	int skipTableSize = 126 * sizeof(int);
	int mapSize = inputObj.getMapSize();
    int lineDataSize = inputObj.getLineDataSize();

	char* d_fullText;
	char* d_testPattern;
	int* d_skipTable;
	unsigned int* d_numMatches;
	int* d_map;
	int* d_lineData;
    int * d_output;

	cudaMalloc((void**)& d_fullText, fullTextSize);
	cudaMalloc((void**)& d_testPattern, patternSize);
	cudaMalloc((void**)& d_skipTable, skipTableSize);
	cudaMalloc((void**)& d_numMatches, sizeof(unsigned int));
	cudaMalloc((void**)& d_map, mapSize);
    cudaMalloc((void**)& d_lineData, lineDataSize);


	cudaMemcpy(d_fullText, flatText, fullTextSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_testPattern, testPattern, patternSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_skipTable, skipTable, skipTableSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_numMatches, numMatches, sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_map, map, mapSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_lineData, lineData, lineDataSize, cudaMemcpyHostToDevice);

    int numBlocks = determineNumBlocks(inputObj.getChunks());
    cudaMalloc((void**)& d_output, numBlocks * sizeof(int));
    int * output = (int *)malloc(sizeof(int) * numBlocks);
    for(int i = 0; i < numBlocks; i++){
        output[i] = 0;
    }
    cudaMemcpy(d_output, output, numBlocks * sizeof(int), cudaMemcpyHostToDevice );

    time_t start, end , start1,end1 = 0;
    int text_len = strlen(flatText);
    int pat_len = strlen(testPattern); 
    int num_chunks = inputObj.getChunks().size(); 
    cudaDeviceSynchronize();
    //time(&start);   
    start = clock();

	horspool_match << <numBlocks, NUM_THREADS_PER_BLOCK, NUM_THREADS_PER_BLOCK * sizeof(int) >> > (d_fullText, d_testPattern, d_skipTable, d_numMatches, CHUNK_SIZE, 
        d_map, d_lineData, num_chunks , text_len, pat_len, d_output);
        cudaDeviceSynchronize();

    end = clock();

    
    start1 = clock();
       unsigned int result = 0;
    for(int myId =0; myId < numBlocks * NUM_THREADS_PER_BLOCK; myId++){
        result += linear_horspool_match(flatText, testPattern, skipTable, numMatches, CHUNK_SIZE, 
            map, lineData, num_chunks , text_len, pat_len, myId);    
    }
    end1 = clock();
    //cout << "result is " << result << endl;
    *numMatches = result;
    cudaDeviceSynchronize();
    end = clock(); 

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


	cudaFree(d_fullText); cudaFree(d_testPattern); cudaFree(d_skipTable); cudaFree(d_numMatches); cudaFree(d_map); cudaFree(d_lineData);
	

	free(testPattern);
	free(skipTable);
	free(numMatches);

    /*int num_matches = 0;
    int* occ = horspool_match (test_str, 0, strlen(test_str), test_pattern, strlen(test_pattern),
        skip, &num_matches);
    // printf("Occurences of %s: %d\n", test_pattern, occ);

    int line_start, line_end;
    int pat_len = strlen(test_pattern);
    printf("%d matches found!\n", num_matches);
    for (int i = 0; i < num_matches; ++i) {
        printf("\n");
        // printf("Pattern found at index %d!\n", occ[i]);

        // get inclusive line start
        line_start = get_line_start(test_str, occ[i]);
        
        // get exclusive line end
        line_end = get_line_end(test_str, occ[i], pat_len);

        // printf("Line start: %d, end: %d\n", line_start, line_end);
        // print line that pattern was found at
        print_line (test_str, line_start, line_end, occ[i], pat_len);
    }
    printf("\n");*/
}

int linear_horspool_match (char* text, char* pattern, int* shift_table, unsigned int* num_matches, int chunk_size,
    int* map, int* lineData, int num_chunks, int text_size, int pat_len, int myId) {

        int count = 0;
        int lineDataIdx = map[myId];
        int num_entries = lineData[lineDataIdx];
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
 *    map          {int*}:
 *    lineData     {int*}:
 *    num_chunks    {int}: Total number of chunks
 *    text_size     {int}: Integer text length
 *    pat_len       {int}: Integer pattern length
 *  Returns:
 *    None
 */ 
 
 /* lineData data structure
struct line_break_entry {
	int line_break_index;
	int line_break_number;
}

struct data_point {
	int num entries;
	line_break_entry[num_entries];
}
*/

 __global__ void horspool_match (char* text, char* pattern, int* shift_table, unsigned int* num_matches, int chunk_size,
    int* map, int* lineData, int num_chunks, int text_size, int pat_len , int * d_output) {
    extern __shared__ int s[];


    int count = 0;
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    if(myId > num_chunks){ //if thread is an invalid thread
        return;
    }
    int lineDataIdx = map[myId];
    int num_entries = lineData[lineDataIdx];
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
    atomicAdd(num_matches, count);
    /*s[threadIdx.x] = count;
    __syncthreads();
    // Add count to total matches atomically
    if (threadIdx.x == 0 ){
        int sum = 0; 
        for(int idx =0; idx < NUM_THREADS_PER_BLOCK; idx++){
            sum += s[idx];
        }
        d_output[blockIdx.x] = sum;
    }
    */
    

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

/**
*  Purpose:
*    Get text line start index from pattern index
*  
*  Args:
*    text       {char*}: Text c-string
*    idx          {int}: Pattern start index
*  
*  Returns:
*    {int}: Inclusive line start index  
*/ 
int get_line_start (char* text, int idx)
{
    const char NEW_LINE = '\n';

    int start_idx = idx;

    while (start_idx != 0 && text[start_idx] != NEW_LINE) {
        // decrement until new line or text start reached
        --start_idx;
    }

    return start_idx;
}

/**
*  Purpose:
*    Get text line end index from pattern index
*  
*  Args:
*    text       {char*}: Text c-string
*    idx          {int}: Pattern start index
*    pattern_len  {int}: Optional param, pass if you know pattern length
*  
*  Returns:
*    {char*}: Exclusive line end index 
*/ 
int get_line_end (char* text, int idx, int pattern_len)
{
    const char NULL_TERM = '\0';
    const char NEW_LINE = '\n';

    int end_idx = idx + pattern_len - 1;

    while (text[end_idx] != NULL_TERM && text[end_idx] != NEW_LINE) {
        // Increment until new line or null terminator found in text
        ++end_idx;
    }

    return end_idx;
}

/**
*  Purpose:
*    Print c-string substring using indices with pattern highlighted in red
* 
*  Args:
*    text      {char*}: Target c-string
*    start_idx   {int}: Inclusive substring start index
*    end_idx     {int}: Exclusive substring end index
*    pat_start   {int}: Index of first pattern character
*    pat_len     {int}: Pattern length
* 
*  Returns:
*    None
*/ 

#define ANSI_COLOR_RED "\x1b[31m"
#define ANSI_COLOR_RESET "\x1b[0m"

void print_line (char* text, int start_index, int end_index, int pat_start, int pat_len) 
{
    int is_red = 0;
    int pat_end = pat_start + pat_len; 

    for (int i = start_index; i < end_index; ++i) {
        if(i == pat_start) {
        // apply red highlight to pattern
        printf(ANSI_COLOR_RED);
        is_red = 1;
        } else if (i == pat_end) {
        // remove red highlight
        printf(ANSI_COLOR_RESET);
        }
        printf("%c", text[i]);
    }

    if (is_red) {
        // Remove highlight if still applied by line end
        printf(ANSI_COLOR_RESET);
    }
}


