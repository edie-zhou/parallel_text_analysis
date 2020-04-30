#pragma once

#include <stdio.h>
#include <malloc.h>
#include <time.h>
#include <errno.h>
#include <iostream>
#include <vector>

#define CHUNK_SIZE 256
//#define CHUNK_SIZE 6

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

	char** getFullText() const { return cStyleArrStrings; }
	char* flattenText();

	vector<string_chunk> getChunks() const { return chunks; }
	int* getMap() const { return map; }
	int* getLineData() const { return lineData; }
	char* getLineDataResponse() const { return lineDataResponse; }

	int getTextSize() const { return textSize; }
	int getMapSize() const { return mapSize; }
	int getLineDataSize() const { return lineDataSize; }
	int getLineDataResponseSize() const { return lineDataResponseSize; }
	int getNumLines() const { return numLineBreaks; }

private:

	bool flattenedTextBool;
	int textSize;

	int numLineBreaks;
	int mapSize;
	int lineDataSize;
	int lineDataResponseSize;

	const char* filename;
	char** cStyleArrStrings;
	char* flattenedText;

	int* map;
	int* lineData;
	char* lineDataResponse;

	//used when allocating data for GPU
	int stringCount;
	vector<string_chunk> chunks;
	vector<int> globalIndices;			//array of new line indices for entire string

	void splitOnChars();
	void array_from_chunk_vector();

	void createLineData();
	void createLineDataResponse();

	void clean();

};
