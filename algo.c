/**
 *  algo.c
 *  Single thread implementation of boyer-moore
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const int ASCII_OFF = 32;

int horspool_match (char* pattern, char* text);
int* create_shifts (char* pattern);

/**
 *  Driver function
 */
int main(int argc, char* argv[])
{
  
  char* test_str = "~~,  <-----";
  char* test_pattern = "~~,  ";
  int occ = horspool_match(test_pattern, test_str);
  printf("Occurences of %s: %d\n", test_pattern, occ);
  return 0;
}

/**
 *  Purpose:
 *    Boyer-Moore-Horspool pattern matching algorithm implementation
 * 
 *  Args:
 *    pattern {char*}: desired pattern c-string 
 *    text    {char*}: text c-string
 * 
 *  Returns:
 *    {int}: Number of found pattern instances
 */ 
int horspool_match (char* pattern, char* text)
{
  // Offset for first ASCII character


  int num_occ = 0;

  int* table = create_shifts (pattern);
  int pat_length = strlen(pattern);
  int text_length = strlen(text);

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

  return num_occ;
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


