/**
 *  algo.c
 *  Single thread implementation of boyer-moore
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int horspool_match (char* pattern, char* text);
int* create_shifts (char* pattern);

/**
 *  Driver function
 */
int main(int argc, char* argv[])
{
  char* test_str = "under the nut butt hut";
  char* test_pattern = "ABARB";
  int occ = horspool_match(test_pattern, test_str);
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
  const int ASCII_OFF = 32;

  int num_occ;

  int* table = create_shifts (pattern);

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
    shift_table[pattern[j] - 32] = length - 1 - j;
  }

  return shift_table;
}


