/*
You can compile this project by running:

$ make

You can run the binary files with:

$ ./mpi
$ ./openmp
$ ./openmpC
*/

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ROWS 1000
#define COLUMNS 1000
#define DEG_FREEDOM 4
#define SEARCH_MAX_SIZE 6

void generate_matrix(int print);
int in_border(int row, int col);
int check_pal_size_n_from_cell(int palindrome_size, int cell_row, int cell_col,
                               int row_direction, int col_direction);
int count_pal_all_dir(int pal_size, int cell_row, int cell_col);
char matrix[ROWS][COLUMNS];
int degree_of_freedom_row[DEG_FREEDOM] = {
    0, 1, 1, 1}; // right_left, up_down, diagonally up_down_left, diagonally
                 // up_down_right
int degree_of_freedom_col[DEG_FREEDOM] = {
    -1, 0, -1, 1}; // right_left, up_down, diagonally up_down_left, diagonally
                   // up_down_right

int main() {
  printf("This Calculation may take up to 1 minutes and nothing printed. \n "
         "Because all outputs will returned to Master(rank0) machine and then "
         "Printed.\n\n");
  generate_matrix(0);
  int processor_count = omp_get_num_procs();
  omp_set_num_threads(processor_count);

  for (int t = 0; t < processor_count; t++) {
    for (int k = 1; k <= SEARCH_MAX_SIZE; k++) {
      double begin = omp_get_wtime();
      int cnt = 0;
      int th_num = -1;
#pragma omp parallel for num_threads(t + 1) reduction(+ : cnt)
      for (int i = 0; i < ROWS; i++) {
        th_num = omp_get_num_threads();
        for (int j = 0; j < COLUMNS; j++) {
          cnt += count_pal_all_dir(k, i, j);
        }
      }
      double end = omp_get_wtime();
      double time_spend = end - begin;

      printf("%d palindromes of size %d found in %f s. using %d Thread(s)\n",
             cnt, k, time_spend, th_num);
    }
    printf("*************************************************\n");
  }
}

void generate_matrix(int print) {
  srand(time(NULL));
  for (int i = 0; i < ROWS; i++) {
    for (int j = 0; j < COLUMNS; j++) {
      matrix[i][j] = (rand() % 26) + 'A';
      if (print)
        printf("%c ", matrix[i][j]);
    }
    if (print)
      printf("\n");
  }
}

int in_border(int row, int col) {
  if (row >= 0 && col >= 0 && row < ROWS && col < COLUMNS)
    return 1;
  return 0;
}

int count_pal_all_dir(int pal_size, int cell_row, int cell_col) {
  int cnt = 0;
  for (int i = 0; i < DEG_FREEDOM; i++) {
    if (check_pal_size_n_from_cell(pal_size, cell_row, cell_col,
                                   degree_of_freedom_row[i],
                                   degree_of_freedom_col[i]))
      cnt++;
  }

  return cnt;
}

int check_pal_size_n_from_cell(int palindrome_size, int cell_row, int cell_col,
                               int row_direction, int col_direction) {
  char st[100000] = {};
  int st_ind = -1;
  int half_size;

  if ((palindrome_size & 1) == 0)
    half_size = palindrome_size / 2;
  else
    half_size = (palindrome_size / 2) + 1;

  int reached_mid = 0;
  for (int i = 0; i < palindrome_size; i++) {
    if (!in_border(cell_row, cell_col))
      return 0;

    if (reached_mid) {
      if (st[st_ind] != matrix[cell_row][cell_col])
        return 0;
      st_ind--;
      cell_row += row_direction;
      cell_col += col_direction;
      continue;
    }

    if (st_ind + 1 != half_size) {
      st_ind++;
      st[st_ind] = matrix[cell_row][cell_col];
    }
    if (st_ind + 1 == half_size) {
      reached_mid = 1;
      if ((palindrome_size & 1) == 1)
        st_ind--;
    }
    cell_row += row_direction;
    cell_col += col_direction;
  }

  return 1;
}
