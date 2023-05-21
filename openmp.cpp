/*
You can compile this project by running:

$ make

You can run the binary files with:

$ ./mpi
$ ./openmp
$ ./openmpC
*/

#include <iostream>
#include <omp.h>
#include <queue>
#include <stack>
#include <string>
#include <thread>
#include <vector>

using namespace std;

#define ROWS 1000
#define COLUMNS 1000
#define DEG_FREEDOM 4
#define SEARCH_MAX_SIZE 6

struct Answer {
  int palindrome_count;
  int palindrome_size;
  double time_spent;
  int th_num;
};

struct Compare {
  bool operator()(const Answer &lhs, const Answer &rhs) {
    if (lhs.th_num == rhs.th_num)
      return lhs.palindrome_size > rhs.palindrome_size;
    return lhs.th_num > rhs.th_num;
  }
};

void generate_matrix(bool print);
bool in_border(int row, int col);
bool check_pal_size_n_from_cell(int palindrome_size, int cell_row, int cell_col,
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
  cout << "This Calculation may take up to 1 minutes and nothing printed. \n "
          "Because all outputs will returned to Master(rank0) machine and then "
          "Printed.\n"
       << endl;
  generate_matrix(false);
  int processor_count = omp_get_num_procs();
  omp_set_num_threads(processor_count);
  priority_queue<struct Answer, std::vector<Answer>, Compare> anspq = {};

  for (int t = 0; t < processor_count; t++) {
    for (int k = 1; k <= SEARCH_MAX_SIZE; k++) {
      struct Answer ans;
      ans.palindrome_size = k;
      double begin = omp_get_wtime();
      int cnt = 0;
#pragma omp parallel for num_threads(t + 1) reduction(+ : cnt)
      for (int i = 0; i < ROWS; i++) {
        ans.th_num = omp_get_num_threads();
        for (int j = 0; j < COLUMNS; j++) {
          cnt += count_pal_all_dir(ans.palindrome_size, i, j);
        }
      }
      double end = omp_get_wtime();
      ans.palindrome_count = cnt;
      ans.time_spent = end - begin;
      anspq.push(ans);
    }
  }

  int last_thnum = 1;
  while (!anspq.empty()) {
    struct Answer tmp = anspq.top();

    if (tmp.th_num != last_thnum) {
      cout << "*************************************************" << endl;
      last_thnum = tmp.th_num;
    }
    cout << fixed << tmp.palindrome_count << " palindromes of size "
         << tmp.palindrome_size << " found in " << tmp.time_spent
         << " s. using " << tmp.th_num << " Thread(s)" << endl;
    anspq.pop();
  }
}

void generate_matrix(bool print) {
  srand(time(NULL));
  for (int i = 0; i < ROWS; i++) {
    for (int j = 0; j < COLUMNS; j++) {
      matrix[i][j] = (rand() % 26) + 'A';
      if (print)
        cout << matrix[i][j] << " ";
    }
    if (print)
      cout << endl;
  }
}

bool in_border(int row, int col) {
  if (row >= 0 && col >= 0 && row < ROWS && col < COLUMNS)
    return true;
  return false;
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

bool check_pal_size_n_from_cell(int palindrome_size, int cell_row, int cell_col,
                                int row_direction, int col_direction) {
  stack<char> st = {};
  int half_size;

  if ((palindrome_size & 1) == 0)
    half_size = palindrome_size / 2;
  else
    half_size = (palindrome_size / 2) + 1;

  bool reached_mid = false;
  for (int i = 0; i < palindrome_size; i++) {
    if (!in_border(cell_row, cell_col))
      return false;

    if (reached_mid) {
      if (st.top() != matrix[cell_row][cell_col])
        return false;
      st.pop();
      cell_row += row_direction;
      cell_col += col_direction;
      continue;
    }

    if (st.size() != half_size) {
      st.push(matrix[cell_row][cell_col]);
    }
    if (st.size() == half_size) {
      reached_mid = true;
      if ((palindrome_size & 1) == 1)
        st.pop();
    }
    cell_row += row_direction;
    cell_col += col_direction;
  }

  return true;
}