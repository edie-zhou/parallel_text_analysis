/**
 *  algo.c
 *  Single thread implementation of boyer-moore
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int* horspool_match (char* pattern, char* text, int* num_matches);
int* create_shifts (char* pattern);
int get_line_start (char* text, int idx);
int get_line_end (char* text, int idx, int pattern_len);
void print_line (char* text, int start_index, int end_index);

/**
 *  Driver function
 */
int main(int argc, char* argv[])
{
  
  char* test_str = "test\nno string here\nm\natestatest\ntest";
  char* test_pattern = "test";
  int num_matches = 0;
  int* occ = horspool_match(test_pattern, test_str, &num_matches);
  // printf("Occurences of %s: %d\n", test_pattern, occ);
  
  int line_start, line_end;
  int pat_length = strlen(test_pattern);
  printf("%d matches found!\n", num_matches);
  for (int i = 0; i < num_matches; ++i) {
    printf("\n");
    // printf("Pattern found at index %d!\n", occ[i]);

    // get inclusive line start
    line_start = get_line_start(test_str, occ[i]);
    
    // get exclusive line end
    line_end = get_line_end(test_str, occ[i], pat_length);

    // printf("Line start: %d, end: %d\n", line_start, line_end);
    // print line that pattern was found at
    print_line (test_str, line_start, line_end);
    
  }
  printf("\n");
  
  return 0;
}

/**
 *  Purpose:
 *    Boyer-Moore-Horspool pattern matching algorithm implementation
 * 
 *  Args:
 *    pattern     {char*}: desired pattern c-string 
 *    text        {char*}: text c-string
 *    num_matches {char*}: Pointer to number of matches found
 * 
 *  Returns:
 *    {int*}: Array of found pattern indices
 */ 
int* horspool_match (char* pattern, char* text, int* num_matches)
{
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
      
      // increment result index, number of matches found, increment text index
      ++idx;
      ++(*num_matches);
      ++i;
      
    } else {
      i = i + table[text[i]];
    }
  }

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
  // const int ASCII_OFF = 32;

  // Line break ASCII value
  const char LINE_BREAK = '\n';

  // Printable ASCII chars are 32-126 inclusive, line break is 10
  const int TABLE_SIZ = 126;

  int length = strlen(pattern);
  int* shift_table = (int*) malloc (sizeof(int) * TABLE_SIZ);
  
  for(int i = 0; i < TABLE_SIZ; i++) {
    // set all entries to longest shift (pattern length)
    shift_table[i] = length;
  }
  for(int j = 0; j < length - 1; j++) {
    // set pattern characters to shortest shifts
    shift_table[pattern[j]] = length - 1 - j;
  }

  // assign shift of 1 for line breaks
  shift_table[LINE_BREAK] = 1;
 
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
    // decrement until new line or text start reached
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
 *    {char*}: Exclusive line end index 
 */ 
int get_line_end (char* text, int idx, int pattern_len)
{
  const char NULL_TERM = '\0';
  const char NEW_LINE = '\n';

  int end_idx = idx + pattern_len - 1;

  while (text[end_idx] != NULL_TERM && text[end_idx] != NEW_LINE) {
    // Increment until new line or null terminator found in text
    ++end_idx;
  }

  return end_idx;
}

/**
 *  Purpose:
 *    Print c-string substring using indices
 * 
 *  Args:
 *    text      {char*}: Target c-string
 *    start_idx   {int}: Inclusive substring start index
 *    end_idx     {int}: Exclusive substring end index
 * 
 *  Returns:
 *    None
 */ 
void print_line (char* text, int start_index, int end_index) 
{
  for (int i = start_index; i < end_index; ++i) {
    printf("%c", text[i]);
  }
}


