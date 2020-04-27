#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "input.h"

#define NUM_THREADS_PER_BLOCK 512

__global__ void horspool_match(char* text, char* pattern, int* shift_table, unsigned int* num_matches, int chunk_size) {
	if (blockIdx.x == 0 && threadIdx.x == 0) {
		*num_matches = 0;
	}
}

using namespace std;

const int ASCII_OFF = 32;
const int chunkSize = 512;

long * calcIndexes(long num_strings, long length){
    long * indexArray = ( long *)malloc(sizeof( long) * (num_strings+2));

    long i;
    long sum ;
   for (sum=0, i=0; i < num_strings; i++, sum+=chunkSize){
       indexArray[i] = sum;
   }
   indexArray[i] = sum + (length - ((num_strings-1) * chunkSize));
   indexArray[++i] = NULL;
   return indexArray;
}

/**
 *  algo.c
 *  Single thread implementation of boyer-moore
 */

/*int* horspool_match (char* text, int txt_start, int txt_end, char* pattern, int pat_len,
int* skip, int* num_matches);*/
int* create_shifts (char* pattern);
int get_line_start (char* text, int idx);
int get_line_end (char* text, int idx, int pattern_len);
void print_line (char* text, int start_index, int end_index, int pat_start, int pat_len);

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
    int* skipTable = create_shifts(testPattern);
	unsigned int* numMatches = (unsigned int*)malloc(1 * sizeof(unsigned int));
	*numMatches = 0;
	int fullTextSize = inputObj.getChunks().size() * CHUNK_SIZE * sizeof(char);
	int patternSize = strlen(testPattern) * sizeof(char);
	int skipTableSize = strlen(testPattern) * sizeof(int);

	char* d_fullText;
	char* d_testPattern;
	int* d_skipTable;
	unsigned int* d_numMatches;

	cudaMalloc((void**)& d_fullText, fullTextSize);
	cudaMalloc((void**)& d_testPattern, patternSize);
	cudaMalloc((void**)& d_skipTable, skipTableSize);
	cudaMalloc((void**)& d_numMatches, sizeof(unsigned int));

	cudaMemcpy(d_fullText, flatText, fullTextSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_testPattern, testPattern, patternSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_skipTable, skipTable, skipTableSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_numMatches, numMatches, sizeof(unsigned int), cudaMemcpyHostToDevice);

	int numBlocks = determineNumBlocks(inputObj.getChunks());
	horspool_match << <numBlocks, NUM_THREADS_PER_BLOCK >> > (d_fullText, d_testPattern, d_skipTable, d_numMatches, CHUNK_SIZE);

	cudaMemcpy(numMatches, d_numMatches, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	
	cudaFree(d_fullText); cudaFree(d_testPattern); cudaFree(d_skipTable); cudaFree(d_numMatches);
	
	cout << "Number of Matches: " << *numMatches << endl;

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

    return 0;
}

/**
*  Purpose:
*    Boyer-Moore-Horspool pattern matching algorithm implementation
* 
*  Args:
*    text        {char*}: text c-string 
*    index_array  {int*}: Array of integer offsets in text
*    pattern     {char*}: desired pattern c-string
*    pat_len       {int}: pattern c-string length
*    skip         {int*}: skip table
*    num_matches {char*}: Pointer to number of matches found
*    d_out        {int*}: Output array of found pattern indices
* 
*  Returns:
*    {int*}: 
*/ 
/*int* horspool_match (char* text, int index_array, int txt_end, char* pattern, int pat_len,
    int* skip, int* num_matches, int* d_out)
{
    int* result = (int*) malloc(0);
    int idx = 0;
    int size = 0;

    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    if(myId > num_strings[0]){
        return;
    }

    int text_length = index_array[myId + 1];

    // don't need to check first pattern_length - 1 characters
    int i = index_array[myId] + pat_len - 1;
    int k = 0;
    while(i < text_length) {
        // reset matched character count
        k = 0;
        while(k <= pat_len - 1 && pattern[pat_len - 1 - k] == text[i - k]) {
            // increment matched character count
            k++;
        }
        if(k == pat_len) {
            // store result index, rellocate result array
            size = ((idx) * sizeof(int)) + sizeof(int);
            result = (int*) realloc(result, size);
            result[idx] = i - pat_len + 1;
            
            // number of matches found, increment result index, increment text index
            ++idx;
            ++i;
        
        } else {
            i = i + skip[text[i]];
        }
    }

    // Add to number of matches found
    *num_matches += idx;
    d_out = result;
//    return result;
	return NULL;
}*/

/**
*  Purpose:
*    Create shift table for Boyer-Moore-Horspool algorithm
*  
*  Args:
*    pattern {char*}: desired pattern c-string
*/ 
int* create_shifts (char* pattern)
{

    int length = strlen(pattern);
    int* shift_table = (int*) malloc (sizeof(int) * length);

    for(int i = 0; i < length; i++) {
        // set all entries to longest shift (pattern length)
        shift_table[i] = length;
    }
	int decrement = 1;
	for (int i = 0; i < length - 1; i++) {
		shift_table[i] -= decrement;
		decrement++;
	}

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


