#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

const int ASCII_OFF = 32;

char * readFile (char * filename, long * length){
    char * buffer = 0;
    FILE * f = fopen (filename, "rb");

    if (f)
    {
    fseek (f, 0, SEEK_END);
    *length = ftell (f);
    fseek (f, 0, SEEK_SET);
    buffer = malloc (*length);
    if (buffer)
    {
        fread (buffer, 1, *length, f);
    }
    fclose (f);
    }

return buffer;
}
int stringCount = 0;

char ** splitOnChars(char * filename){
    FILE *fptr;
    char ** res  = NULL;
	fptr=fopen("sample-texts/J. K. Rowling - Harry Potter 3 - Prisoner of Azkaban.txt","r");
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


__global__ void horspool_match (char** input, int* table, int * d_output, int* pattern_length, int * num_strings, char * pattern)
{
  // Offset for first ASCII character


  int num_occ = 0;
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  if(myId > num_strings[0]){
      return;
  }
  char * text = input[myId];
  int pat_length = pattern_length[0];
  int text_length = 512;


  int i = pat_length - 1;
  int k = 0;
  while(i < text_length) {
    k = 0;
    while(k <= pat_length - 1 && pattern[pat_length - 1 - k] == text[i - k]) {
      k++;
    }
    if(k == pat_length) {
      num_occ++;
      i++;
      //i = i - pat_length  + 1;
    } else {
      i = i + table[text[i] - ASCII_OFF];
    }
  }

// Test pattern table
// const int TABLE_SIZ = 95;
// for (int i = 0; i < TABLE_SIZ; ++i) {
//   char entry = i + ASCII_OFF;
//   printf("char: %c, offset: %d\n", entry, table[i]);
// }

atomicAdd(&d_output[0], num_occ);
}

/**
 *  Purpose:
 *    Create shift table for Boyer-Moore-Horspool algorithm
 *  
 *  Args:
 *    pattern {char*}: desired pattern c-string
 */ 
 int * create_shifts (char* pattern)
{
  // Printable ASCII chars are 32-126 inclusive
  const int TABLE_SIZ = 95;
  
  int length = strlen(pattern);
  int* shift_table = (int*) malloc (sizeof(int) * TABLE_SIZ);
  
  for(int i = 0; i < TABLE_SIZ; i++) {
    shift_table[i] = length;
  }
  for(int j = 0; j < length - 1; j++) {
    shift_table[pattern[j] - ASCII_OFF] = length - 1 - j;
  }

  return shift_table;
}


int main (void)
{
    char ** input = splitOnChars(NULL);
    
    char * pattern = "call";
    const int maxThreadsPerBlock = 50;
    int threads = maxThreadsPerBlock;
    int blocks = stringCount / maxThreadsPerBlock;
    blocks++;
    int sizeOfString = 512 * sizeof(char);
    int sizeOfInput = sizeOfString * stringCount;


    int* table = create_shifts (pattern);
    int * d_table;
    char ** d_input;
    int * d_output;
    int * outpt = (int *)malloc(sizeof(int) *1);
    int * d_num_strings;
    int patternLength = strlen(pattern);
    int * d_patt_length;
    char * d_pattern;

	cudaMalloc((void **)&d_table, 95 * sizeof(int));
    cudaMalloc((void **)&d_input ,sizeOfInput);
    cudaMalloc((void **)&d_output, sizeof(int) *1);
    cudaMalloc((void **)&d_patt_length, sizeof(int) *1);
    cudaMalloc((void **)&d_num_strings, sizeof(int) *1);
    cudaMalloc((void **)&d_pattern, 1+patternLength);

    cudaMemcpy(d_input, input, sizeOfInput, cudaMemcpyHostToDevice);
    char * tempString
    for(int i =0; input[i] != NULL; i++){
        cudaMalloc((void **)&tempString, sizeOfString);
        cudaMemcpy(&d_input[i], &input[i],sizeOfString, cudaMemcpyHostToDevice);
    }
    cudaMemcpy(d_table, table, sizeof(int) * 95, cudaMemcpyHostToDevice);
    cudaMemcpy(d_patt_length, &patternLength, sizeof(int) * 1, cudaMemcpyHostToDevice);    
    cudaMemcpy(d_num_strings, &stringCount, sizeof(int) * 1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pattern, &pattern, patternLength+1, cudaMemcpyHostToDevice);

    horspool_match<<<blocks,threads>>>(d_input, d_table, d_output,d_patt_length, d_num_strings, d_pattern);
    cudaMemcpy(outpt, d_output, sizeof(int) * 1, cudaMemcpyDeviceToHost);

    printf("%d number of occurences\n", outpt[0]);

    return 212;
}
