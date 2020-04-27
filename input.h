#include <stdio.h>
#include <malloc.h>
#include <time.h>
#include <vector>

#define CHUNK_SIZE 512

struct string_chunk {
	char* str;
	vector<int> newLineIndices;		//index of new lines in this string
	vector<int> lineNumbers;		//global rank of each new line in this string
};

using namespace std;

//used when allocating data for GPU
int stringCount;
vector<string_chunk> chunks;
vector<int> globalIndices;			//array of new line indices for entire string

char ** splitOnChars(char * filename);
