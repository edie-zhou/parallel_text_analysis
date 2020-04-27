#pragma once

#include "input.h"

using namespace std;

Input::Input() {
	filename = "sample-texts/J. K. Rowling - Harry Potter 3 - Prisoner of Azkaban.txt";
	flattenedText = false;
	splitOnChars();
}

Input::Input(const char* fname) {
	filename = fname;
	flattenedText = false;
	splitOnChars();
}

Input::~Input() {
	clean();
}

char* Input::flattenText() {
	flattenedText = (char*)malloc(chunks.size() * CHUNK_SIZE * sizeof(char));
	for (int row = 0; row < chunks.size(); row++) {
		for (int col = 0; col < CHUNK_SIZE; col++) {
			flattenedText[CHUNK_SIZE * row + col] = cStyleArrStrings[row][col];
		}
	}
	flattenedTextBool = true;
	return flattenedText;
}

void Input::splitOnChars() {
	stringCount = 0;
	FILE* fptr;
	fptr = fopen(filename, "r");
	char ch;
	while ((ch = fgetc(fptr)) != EOF) {
		string_chunk chunk;
		chunk.str = (char*)malloc(512 * sizeof(char));
		chunk.str[0] = ch;
		if (ch == '\n') {
			globalIndices.push_back(chunks.size() * CHUNK_SIZE);
			chunk.newLineIndices.push_back(0);
			chunk.lineNumbers.push_back(globalIndices.size());
		}
		for (int count = 1; (count < CHUNK_SIZE && (ch = fgetc(fptr)) != EOF); count++) {
			chunk.str[count] = ch;
			if (ch == '\n') {
				globalIndices.push_back(chunks.size() * CHUNK_SIZE + count);
				chunk.newLineIndices.push_back(count);
				chunk.newLineIndices.push_back(globalIndices.size());
			}
		}
		stringCount++;
		chunks.push_back(chunk);
	}
	array_from_chunk_vector();
	return;
}

void Input::array_from_chunk_vector() {
	cStyleArrStrings = (char**)malloc(chunks.size() * sizeof(char*));
	for (int i = 0; i < chunks.size(); i++) {
		cStyleArrStrings[i] = chunks.at(i).str;
	}
	return;
}

void Input::clean() {
	for (int i = 0; i < chunks.size(); i++) {
		free(chunks.at(i).str);
	}
	free(cStyleArrStrings);
	if (flattenedTextBool) free(flattenedText);
	return;
}
