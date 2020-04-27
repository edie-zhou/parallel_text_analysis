#pragma once

#include <stdio.h>
#include <malloc.h>
#include <time.h>
#include <errno.h>
#include <iostream>
#include <vector>

#define CHUNK_SIZE 512

using namespace std;
struct string_chunk {
	char* str;
	vector<int> newLineIndices;		//index of new lines in this string
	vector<int> lineNumbers;		//global rank of each new line in this string
};

class Input {

public:

	Input();
	Input(const char* fname);
	~Input();

	vector<string_chunk> getChunks() const { return chunks; }
	char** getArrayStrings() const { return cStyleArrStrings; }

private:

	const char* filename;
	char** cStyleArrStrings;

	//used when allocating data for GPU
	int stringCount;
	vector<string_chunk> chunks;
	vector<int> globalIndices;			//array of new line indices for entire string

	void splitOnChars();
	char** array_from_chunk_vector();

	void clean();

};