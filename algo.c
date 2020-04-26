/**
 *  algo.c
 *  Single thread implementation of boyer-moore
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int* horspool_match (char* pattern, char* text);
int* create_shifts (char* pattern);
int get_line_start (char* text, int idx);
int get_line_end (char* text, int idx, int pattern_len);

/**
 *  Driver function
 */
int main(int argc, char* argv[])
{
  
  char* test_str = "test\nme\ntest me please";
  char* test_pattern = "test";
  int* occ = horspool_match(test_pattern, test_str);
  // printf("Occurences of %s: %d\n", test_pattern, occ);
  
  size_t occ_size = sizeof(occ) / sizeof(int);
  for (int i = 0; i < occ_size; ++i) {
    printf("%d\n", occ[i]);
  }

  int pat_length = strlen(test_pattern);
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
 *    {int*}: Array of found pattern indices
 */ 
int* horspool_match (char* pattern, char* text)
{
  // Offset for first ASCII character
  const int ASCII_OFF = 32;
  const char NEW_LINE = '\n';

  int* result = (int*) malloc(0);
  int idx = 0;
  int size = 0;

  int* table = create_shifts(pattern);
  int pat_length = strlen(pattern);
  int text_length = strlen(text);

  int i = pat_length - 1;
  int k = 0;
  while(i < text_length) {
    k = 0;
    while(k <= pat_length - 1 && pattern[pat_length - 1 - k] == text[i - k]) {
      // increment number of matched characters
      k++;
    }
    if(k == pat_length) {
      // store result index, rellocate result array
      size = ((idx) * sizeof(int)) + sizeof(int);
      result = (int*) realloc(result, size);
      result[idx] = i - pat_length + 1;
      ++idx;
      
      // increment text index
      i++;
      while (text[i] == NEW_LINE) {
        // Skip over new lines
        ++i;
      }
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

  return result;
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
  // Offset for first ASCII character
  const int ASCII_OFF = 32;

  // Printable ASCII chars are 32-126 inclusive
  const int TABLE_SIZ = 95;
  
  int length = strlen(pattern);
  int* shift_table = (int*) malloc (sizeof(int) * TABLE_SIZ);
  
  for(int i = 0; i < TABLE_SIZ; i++) {
    // set all entries to longest shift (pattern length)
    shift_table[i] = length;
  }
  for(int j = 0; j < length - 1; j++) {
    // set pattern characters to shortest shifts
    shift_table[pattern[j] - ASCII_OFF] = length - 1 - j;
  }

  return shift_table;
}

/**
 *  Purpose:
 *    Get text line start index from pattern index
 *  
 *  Args:
 *    text       {char*}: Text c-string
 *    idx          {int}: Pattern start index
 *  
 *  Returns:
 *    {int}: Inclusive line start index  
 */ 
int get_line_start (char* text, int idx)
{
  const char NEW_LINE = '\n';

  int start_idx = idx;

  while (start_idx != 0 && text[start_idx] != NEW_LINE) {
    --start_idx;
  }

  return start_idx;
}

/**
 *  Purpose:
 *    Get text line end index from pattern index
 *  
 *  Args:
 *    text       {char*}: Text c-string
 *    idx          {int}: Pattern start index
 *    pattern_len  {int}: Optional param, pass if you know pattern length
 *  
 *  Returns:
 *    {char*}: Inclusive line end index 
 */ 
int get_line_end (char* text, int idx, int pattern_len)
{
  const char NULL_TERM = '\0';
  const char NEW_LINE = '\n';

  int end_idx = idx;

  while (text[end_idx] != NULL_TERM && text[end_idx] != NEW_LINE) {
    // Increment until new line or null terminator
    ++end_idx;
  }

  return end_idx - 1;
}


