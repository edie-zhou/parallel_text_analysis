#include <stdio.h>
#include <malloc.h>
#include <vector>
#include <iostream>


using namespace std;

int stringCount = 0;

char ** splitOnChars(char * filename){
    FILE *fptr;
	fptr=fopen("J. K. Rowling - Harry Potter 3 - Prisoner of Azkaban.txt","r");
    vector<char*> split_array ;  
    char ch;
    while((ch=fgetc(fptr))!=EOF) {
        char * tempString = (char*)malloc(512 * sizeof(char));
        for(int count = 0; (count < 511 && (ch=fgetc(fptr))!=EOF); count++ ){
            if(ch != '\n'){
                tempString[count] = ch;
            }
            else{
                count--;
            }
        }
        tempString[511] = '\0';
        split_array.push_back(tempString);
        stringCount++;
	}
    split_array.push_back(NULL);
    char** array = (char**)&split_array[0];     
return array;
}



int main() {

    char ** result = splitOnChars(NULL);
    for(int i=0; result[i] != NULL; i++){
        cout << result[i] << "\n";
    }
}


