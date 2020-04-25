#include <stdio.h>
#include <malloc.h>
#include <time.h>
     

int stringCount = 0;

char ** splitOnChars(char * filename){
    FILE *fptr;
    char ** res  = NULL;
	fptr=fopen("J. K. Rowling - Harry Potter 3 - Prisoner of Azkaban.txt","r");
    char ch;
    int n_spaces = 0;
    while((ch=fgetc(fptr))!=EOF) {
        char * tempString = (char*)malloc(512 * sizeof(char));
        res = (char **)realloc (res, sizeof (char*) * ++n_spaces);
        if (res == NULL)
            exit (-1); /* memory allocation failed */
        res[n_spaces-1] = tempString;
        tempString[0] = ch;
        for(int count = 1; (count < 511 && (ch=fgetc(fptr))!=EOF); count++ ){
            if(ch != '\n'){
                tempString[count] = ch;
            }
            else{
                count--;
            }
        }
        tempString[511] = '\0';
        stringCount++;
	}
    res = (char **)realloc (res, sizeof (char*) * (n_spaces+1));
    res[n_spaces] = 0;
    return res;
}


int main() {
    clock_t start, end;
    double cpu_time_used;

    start = clock();

    char ** result = splitOnChars(NULL);

    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("%f - total time\n", cpu_time_used);
    for(int i=0; result[i] != NULL; i++){
    printf("%s \n", result[i]);
    }
}


