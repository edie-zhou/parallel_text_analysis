// #pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <iostream>
#include "cuda_runtime.h"
#include <iomanip>      // std::setprecision

#include "device_launch_parameters.h"

#include "input.h"

using namespace std;

#define NUM_THREADS_PER_BLOCK 512

int* create_shifts (char* pattern);

int linear_horspool_match (char* text, char* pattern, int* shift_table, unsigned int* num_matches, int chunk_size,
    int num_chunks, int text_size, int pat_len, int myId);

__global__ void horspool_match (char* text, char* pattern, int* shift_table, unsigned int* num_matches, int chunk_size,
    int num_chunks, int text_size, int pat_len);
__global__ void prescan(int *g_odata, int *g_idata, int n);

int determineNumBlocks(vector<string_chunk> chunks) {
	int numBlocks = 0;
	for (int i = 0; i < chunks.size(); i = i + NUM_THREADS_PER_BLOCK) {
		numBlocks++;
	}
	return numBlocks;
}

/*
 *  Driver function
 *  argv[0] is target pattern string
 *  argv[1] is text path
 */
int main(int argc, char* argv[])
{
    const int TABLE_SIZ = 126;
    int target_len = 0;

    if (argc == 2 && (strcmp(argv[1], "-h") || strcmp(argv[1], "--help"))){
        cout << "`match.exe` finds exact matches to a target string in text files." << endl
            << "Type ./main.exe {target_string} {text file path} to use the program." << endl
            << "Text file paths must be relative to the directory of `main.exe`." << endl;
        exit(0);
    } else if (argc == 1) {
        cout << "ERROR: Please pass in a target string and a file path." << endl;
        exit(-1);
    }
    for (int i = 1; i < argc - 1; ++i) {
        target_len += strlen(argv[i]);
    }
    target_len += argc - 3;
    

	Input inputObj(argv[argc - 1]);
    char* flatText = inputObj.flattenText();

    char* testPattern = (char*)malloc(target_len * sizeof(char) + 1);
    string input = argv[1];
    if (argc > 3) {
        for (int i = 2; i < argc - 1; ++i) {
            input = input + " " + argv[i];

        }
    }

    strcpy (testPattern, input.data());
    testPattern[target_len] = '\0';

    int* skipTable = create_shifts(testPattern);
	unsigned int* numMatches = (unsigned int*)malloc(1 * sizeof(unsigned int));
	*numMatches = 0;

	int fullTextSize = inputObj.getChunks().size() * CHUNK_SIZE * sizeof(char);
	int patternSize = strlen(testPattern) * sizeof(char);
	int skipTableSize = TABLE_SIZ * sizeof(int);

	char* d_fullText;
	char* d_testPattern;
	int* d_skipTable;
	unsigned int* d_numMatches;
    unsigned int* parallel_result = (unsigned int*) malloc(sizeof(unsigned int));

	cudaMalloc((void**)& d_fullText, fullTextSize);
	cudaMalloc((void**)& d_testPattern, patternSize);
	cudaMalloc((void**)& d_skipTable, skipTableSize);
	cudaMalloc((void**)& d_numMatches, sizeof(unsigned int));

	cudaMemcpy(d_fullText, flatText, fullTextSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_testPattern, testPattern, patternSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_skipTable, skipTable, skipTableSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_numMatches, numMatches, sizeof(unsigned int), cudaMemcpyHostToDevice);

    
    time_t start, end, start1, end1 = 0;
    int text_len = strlen(flatText);
    int pat_len = strlen(testPattern); 
    int num_chunks = inputObj.getChunks().size();
    int numBlocks = determineNumBlocks(inputObj.getChunks());
    cudaDeviceSynchronize();

    time(&start);   
    start = clock();

	horspool_match << <numBlocks, NUM_THREADS_PER_BLOCK, NUM_THREADS_PER_BLOCK * sizeof(int) >> > (d_fullText, d_testPattern, d_skipTable, d_numMatches, CHUNK_SIZE, 
        num_chunks, text_len, pat_len);
        cudaDeviceSynchronize();
    
    cudaMemcpy(parallel_result, d_numMatches, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    end = clock();
    

    start1 = clock();
    unsigned int result = 0;
    for(int myId =0; myId < numBlocks * NUM_THREADS_PER_BLOCK; myId++){
        result += linear_horspool_match(flatText, testPattern, skipTable, numMatches, CHUNK_SIZE, 
            num_chunks, text_len, pat_len, myId);    
    }
    end1 = clock();
    cudaDeviceSynchronize();

    // Calculating total time taken by the program. 
    double time_taken = double(end - start)/ CLOCKS_PER_SEC; 
    cout << "Time taken by parallel program: " << setprecision(9) << time_taken << " secs." << endl;
    cout << "There are " << *parallel_result << " exact matches to string `" << input << "`" << 
        endl << "found by parallel program in file `" << argv[argc-1] <<"`"<< endl << endl;

    time_taken = double(end1 - start1)/ CLOCKS_PER_SEC;
    cout << "Time taken by linear program: " << setprecision(9) << time_taken <<  " secs." << endl; 
    cout << "There are " << result << " exact matches to string `" << input << "`" <<
        endl << "found by linear program in file `" << argv[argc-1] <<"`"<< endl;

    cudaFree(d_fullText);
    cudaFree(d_testPattern);
    cudaFree(d_skipTable);
    cudaFree(d_numMatches);

	free(testPattern);
	free(skipTable);
    free(numMatches);
}

int linear_horspool_match (char* text, char* pattern, int* shift_table, unsigned int* num_matches, int chunk_size,
    int num_chunks, int text_size, int pat_len, int myId) {
        
        const int TABLE_SIZ = 126;

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

            if (text[i] >= TABLE_SIZ || text[i] < 0) {
                // move to next char if unknown char (Unicode, etc.)
                ++i;
            } else {
                while(k <= pat_len - 1 && pattern[pat_len - 1 - k] == text[i - k]) {
                // increment matched character count
                    k++;
                }
                if(k == pat_len) {
                // increment pattern count, text index
                    ++count;
                    ++i;
        
                } else {
                    // add on shift if known char
                    i = i + shift_table[text[i]];
                }
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
    int num_chunks, int text_size, int pat_len) {
    
    const int TABLE_SIZ = 126;

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

        if (text[i] >= TABLE_SIZ || text[i] < 0) {
            // move to next char if unknown char (Unicode, etc.)
            ++i;
        } else {
            while(k <= pat_len - 1 && pattern[pat_len - 1 - k] == text[i - k]) {
            // increment matched character count
                k++;
            }
            if(k == pat_len) {
            // increment pattern count, text index
                ++count;
                ++i;
    
            } else {
                // add on shift if known char
                i = i + shift_table[text[i]];
            }
        }
    }

    atomicAdd(num_matches, count);
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

    // Printable ASCII chars are 32-126 inclusive, line break is 10
    const int TABLE_SIZ = 126;

    const int FIRST_ASCII = 32;

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

    // assign shift of 1 for unprintable characters
    for (int i = 0; i < FIRST_ASCII; ++i) {
        shift_table[i] = 1;
    }

    return shift_table;
}