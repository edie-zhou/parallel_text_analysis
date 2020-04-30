// #pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <iostream>
#include "cuda_runtime.h"
#include <iomanip>      // std::setprecision

#include "device_launch_parameters.h"
#include <iomanip>

#include "input.h"

using namespace std;

#define NUM_THREADS_PER_BLOCK 512

int* create_shifts (char* pattern);
void print_lines(char* lineDataResponse, int numLines);

int linear_horspool_match (char* text, char* pattern, int* shift_table, unsigned int* num_matches, int chunk_size,
    int num_chunks, int text_size, int pat_len, int myId);

__global__ void horspool_match (char* text, char* pattern, int* shift_table, unsigned int* num_matches, int chunk_size,
    int num_chunks, int text_size, int pat_len, int* map, int* lineData, char* lineDataResponse);
__global__ void prescan(int *g_odata, int *g_idata, int n);
__device__ void addLineResponse(char* text, int textIdx, int chunkBeginIdx, int* lineData, int lineDataIndex, char* lineDataResponse, int myId, int chunkSize, int pat_len);
__device__ int getLocalNewLineIndex(char* text, int textIdx, int chunkBeginIdx, int* chunksToSkip, int chunkSize, int myId);
__device__ void writeString(char* text, int stringIndex, char* lineDataResponse, int lineNumber, int padding, int pat_len);

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

    // printf("%d", argc);
    if (argc == 2 && (strcmp(argv[1], "-h") || strcmp(argv[1], "--help"))){
        cout << "`match.exe` finds exact matches to a target string in text files." << endl
            << "Type ./main.exe {target_string} {text file path} to use the program." << endl
            << "Text file paths must be relative to the directory of `main.exe`." << endl;
        exit(0);
    }
	else if (argc != 3) {
        cout << "ERROR: Please pass in a target string and a file path." << endl;
        exit(-1);
    }

	Input inputObj(argv[2]);
    char* flatText = inputObj.flattenText();

    int input_len = strlen(argv[1]);
	char* testPattern = (char*)malloc(input_len * sizeof(char) + 1);
    testPattern = strcpy(testPattern, argv[1]);
    testPattern[input_len] = '\0';
    int* skipTable = create_shifts(testPattern);
	unsigned int* numMatches = (unsigned int*)malloc(1 * sizeof(unsigned int));
	*numMatches = 0;
	int* map = inputObj.getMap();
	int* lineData = inputObj.getLineData();
	char* lineDataResponse = inputObj.getLineDataResponse();

	int fullTextSize = inputObj.getChunks().size() * CHUNK_SIZE * sizeof(char);
	int patternSize = strlen(testPattern) * sizeof(char);
	int skipTableSize = TABLE_SIZ * sizeof(int);
	int mapSize = inputObj.getMapSize();
	int lineDataSize = inputObj.getLineDataSize();
	int lineDataResponseSize = inputObj.getLineDataResponseSize();

	char* d_fullText;
	char* d_testPattern;
	int* d_skipTable;
	unsigned int* d_numMatches;
    unsigned int* parallel_result = (unsigned int*) malloc(sizeof(unsigned int));
	int* d_map;
	int* d_lineData;
	char* d_lineDataResponse;

	cudaMalloc((void**)& d_fullText, fullTextSize);
	cudaMalloc((void**)& d_testPattern, patternSize);
	cudaMalloc((void**)& d_skipTable, skipTableSize);
	cudaMalloc((void**)& d_numMatches, sizeof(unsigned int));
	cudaMalloc((void**)& d_map, mapSize);
	cudaMalloc((void**)& d_lineData, lineDataSize);
	cudaMalloc((void**)& d_lineDataResponse, lineDataResponseSize);

	cudaMemcpy(d_fullText, flatText, fullTextSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_testPattern, testPattern, patternSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_skipTable, skipTable, skipTableSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_numMatches, numMatches, sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_map, map, mapSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_lineData, lineData, lineDataSize, cudaMemcpyHostToDevice);
    
    time_t start, end , start1,end1 = 0;
    int text_len = strlen(flatText);
    int pat_len = strlen(testPattern); 
    int num_chunks = inputObj.getChunks().size();
    int numBlocks = determineNumBlocks(inputObj.getChunks());
    cudaDeviceSynchronize();

    time(&start);   
    start = clock();

	horspool_match << <numBlocks, NUM_THREADS_PER_BLOCK, NUM_THREADS_PER_BLOCK * sizeof(int) >> > (d_fullText, d_testPattern, d_skipTable, d_numMatches, CHUNK_SIZE, 
        num_chunks, text_len, pat_len, d_map, d_lineData, d_lineDataResponse);
    
	cudaDeviceSynchronize();

	end = clock();
    
	cudaMemcpy(parallel_result, d_numMatches, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy(lineDataResponse, d_lineDataResponse, lineDataResponseSize, cudaMemcpyDeviceToHost);

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
	print_lines(lineDataResponse, inputObj.getNumLines());
    cout << "Time taken by parallel program: " << setprecision(9) << time_taken << endl;
    cout << "There are " << *parallel_result << " exact matches to string `" << argv[1] << "`" << 
        endl << "found by parallel program in file `" << argv[2] <<"`"<< endl << endl;

    time_taken = double(end1 - start1)/ CLOCKS_PER_SEC;
    /*cout << "Time taken by linear program: " << setprecision(9) << time_taken << endl; 
    cout << "There are " << result << " exact matches to string `" << argv[1] << "`" <<
        endl << "found by linear program in file `" << argv[2] <<"`"<< endl;*/

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
    int num_chunks, int text_size, int pat_len, int* map, int* lineData, char* lineDataResponse) {
    
    const int TABLE_SIZ = 126;

    int count = 0;
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    if(myId > num_chunks){ //if thread is an invalid thread
        return;
    }

    int text_length = (chunk_size * myId) + chunk_size + pat_len - 1;

	int mapIdx = myId;
	int numNewLineEntries = lineData[map[mapIdx]];


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
				addLineResponse(text, i, chunk_size * myId, lineData, map[mapIdx], lineDataResponse, myId, chunk_size, pat_len);
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

 __device__ void addLineResponse(char* text, int textIdx, int chunkBeginIdx, int* lineData, int lineDataIndex, char* lineDataResponse, int myId, int chunkSize, int pat_len) {
	 int chunksToSkip = 0;
	 int localNewLineIndex = getLocalNewLineIndex(text, textIdx, chunkBeginIdx, &chunksToSkip, chunkSize, myId);
	 int modifiedLineDataIndex = lineDataIndex;
	 while (chunksToSkip > 0) {
		 int numEntries = lineData[modifiedLineDataIndex];
		 modifiedLineDataIndex++;
		 modifiedLineDataIndex = modifiedLineDataIndex + (numEntries * 2);
		 chunksToSkip--;
	 }
	 int numEntries = lineData[modifiedLineDataIndex];
	 int lineNumber = -1;
	 int offset = -1;
	 for (int i = 0; (i < numEntries) && (lineNumber == -1); i++) {
		 offset += 2;
		 if (lineData[modifiedLineDataIndex + offset] == localNewLineIndex) {
			 lineNumber = lineData[modifiedLineDataIndex + offset + 1];
		 }
	 }

	 int spaceLeft;
	 if ((pat_len % 2) == 0) {
		 spaceLeft = 20 - pat_len;
	 }
	 else {
		 spaceLeft = 20 - (pat_len + 1);
	 }
	 int padding = spaceLeft / 2;
	 writeString(text, textIdx, lineDataResponse, lineNumber, padding, pat_len);
 }

 __device__ int getLocalNewLineIndex(char* text, int textIdx, int chunkBeginIndex, int* chunksToSkip, int chunkSize, int myId) {
	 int tempIdx = textIdx;
	 int globalIndex = -1;
	 while (globalIndex == -1) {
		 if (text[tempIdx] == '\n') {
			 globalIndex = tempIdx;
		 }
		 else {
			 tempIdx++;
		 }
	 }
	 int localIndex = globalIndex - chunkBeginIndex;
	 while (localIndex > chunkSize) {
		 localIndex -= chunkSize;
		 *chunksToSkip = *chunksToSkip + 1;
	 }
	 return localIndex;
 }

 __device__ void writeString(char* text, int stringIndex, char* lineDataResponse, int lineNumber, int padding, int pat_len) {
	 if (lineNumber == -1) return;
	 if (lineDataResponse[(lineNumber - 1) * 21] == '\0') {
		 int startingIndex = stringIndex - pat_len - padding;
		 if (startingIndex < 0) startingIndex = 0;
		 for (int i = 0; i < 20; i++) {
			 lineDataResponse[((lineNumber - 1) * 21) + i] = text[startingIndex];
			 startingIndex++;
		 }
		 lineDataResponse[20] = '\0';
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

void print_lines(char* lineDataResponse, int numLines) {
	cout << "Matches: " << endl << endl;
	for (int i = 0; i < numLines; i++) {
		if (lineDataResponse[21 * i] != '\0') {
			cout << "Line " << i + 1 << ": ";
			for (int j = 0; lineDataResponse[21 * i + j] != '\0'; j++) {
				cout << lineDataResponse[21 * i + j];
			}
			cout << endl;
		}
	}
	cout << endl;
}
