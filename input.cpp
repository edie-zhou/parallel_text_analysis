#include "input.h"

using namespace std;

char** array_from_chunk_vector();

char** splitOnChars(char* filename) {
	stringCount = 0;
	FILE* fptr;
	fptr = fopen("sample_texts/J. K. Rowling - Harry Potter 3 - Prisoner of Azkaban.txt", "r");
	char ch;
	while ((ch = fgetc(fptr)) != EOF) {
		char* tempString = (char*)malloc(512 * sizeof(char));
		string_chunk chunk;
		chunk.str = tempString;
		tempString[0] = ch;
		if (ch == '\n') {
			globalIndices.push_back(chunks.size() * CHUNK_SIZE);
			chunk.newLineIndices.push_back(0);
			chunk.lineNumbers.push_back(globalIndices.size());
		}
		for (int count = 1; (count < CHUNK_SIZE && (ch = fgetc(fptr)) != EOF); count++) {
			tempString[count] = ch;
			if (ch == '\n') {
				globalIndices.push_back(chunks.size() * CHUNK_SIZE + count);
				chunk.newLineIndices.push_back(count);
				chunk.newLineIndices.push_back(globalIndices.size());
			}
		}
		stringCount++;
		chunks.push_back(chunk);
	}
	return array_from_chunk_vector();
}

char** array_from_chunk_vector() {
	char** result = (char**)malloc(chunks.size() * sizeof(char*));
	for (int i = 0; i < chunks.size(); i++) {
		result[i] = chunks.at(i).str;
	}
	return result;
}
