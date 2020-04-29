#pragma once

#include "input.h"

using namespace std;

Input::Input() {
	filename = "sample-texts/J. K. Rowling - Harry Potter 3 - Prisoner of Azkaban.txt";
	flattenedTextBool = false;
	numLineBreaks = 0;
	textSize = 0;
	splitOnChars();
}

Input::Input(const char* fname) {
	filename = fname;
	flattenedTextBool = false;
	numLineBreaks = 0;
	textSize = 0;
	splitOnChars();
}

Input::~Input() {
	clean();
}

char* Input::flattenText() {
	flattenedText = (char*)malloc((textSize + 1) * sizeof(char));
	int count = 0;
	int row;
	int col;
	for (row = 0; row < chunks.size(); row++)
	{
		for (col = 0; col < CHUNK_SIZE; col++) {
			if (CHUNK_SIZE * row + col < textSize) {
				count++;
				flattenedText[CHUNK_SIZE * row + col] = cStyleArrStrings[row][col];
			}
		}
	}
	flattenedText[textSize] = '\0';
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
		chunk.str = (char*)malloc(CHUNK_SIZE * sizeof(char));
		chunk.str[0] = ch;
		textSize++;
		if (ch == '\n') {
			numLineBreaks++;
			globalIndices.push_back(chunks.size() * CHUNK_SIZE);
			chunk.newLineIndices.push_back(0);
			chunk.lineNumbers.push_back(globalIndices.size());
		}
		for (int count = 1; (count < CHUNK_SIZE && (ch = fgetc(fptr)) != EOF); count++) {
			chunk.str[count] = ch;
			textSize++;
			if (ch == '\n') {
				numLineBreaks++;
				globalIndices.push_back(chunks.size() * CHUNK_SIZE + count);
				chunk.newLineIndices.push_back(count);
				chunk.lineNumbers.push_back(globalIndices.size());
			}
		}
		stringCount++;
		chunks.push_back(chunk);
	}
	array_from_chunk_vector();
	createLineData();
	return;
}

void Input::array_from_chunk_vector() {
	cStyleArrStrings = (char**)malloc(chunks.size() * sizeof(char*));
	for (int i = 0; i < chunks.size(); i++) {
		cStyleArrStrings[i] = chunks.at(i).str;
	}
	return;
}

void Input::createLineData() {
	int dataPerNewline = 2;
	lineDataSize = (chunks.size() + (numLineBreaks * dataPerNewline)) * sizeof(int);
	lineData = (int *)malloc(lineDataSize);
	mapSize = chunks.size() * sizeof(int);
	map = (int *)malloc(mapSize);

	int arrIdx = 0;
	for (int i = 0; i < chunks.size(); i++) {
		map[i] = arrIdx;

		lineData[arrIdx] = chunks.at(i).lineNumbers.size();
		int offset = 0;
		for (int j = 0; j < chunks.at(i).lineNumbers.size(); j++) {
			lineData[arrIdx + 1 + offset] = chunks.at(i).newLineIndices.at(j);
			lineData[arrIdx + 1 + offset + 1] = chunks.at(i).lineNumbers.at(j);
			offset += 2;
		}
		arrIdx = arrIdx + 1 + (chunks.at(i).lineNumbers.size() * 2);
	}
}

void Input::clean() {
	for (int i = 0; i < chunks.size(); i++) {
		free(chunks.at(i).str);
	}
	free(cStyleArrStrings);
	if (flattenedTextBool) free(flattenedText);
	free(lineData);
	free(map);
	return;
}
