#include <iostream>
#include <string>
#include <stack>
#include <thread>
#include <omp.h>

using namespace std;

#define ROWS 100
#define COLUMNS 100
#define DEG_FREEDOM 4

void generate_matrix(bool print);
bool in_border(int row, int col);
bool check_pal_size_n_from_cell(int palindrome_size, int cell_row, int cell_col, int row_direction, int col_direction);
int count_pal_all_dir(int pal_size, int cell_row, int cell_col);
char matrix[ROWS][COLUMNS];
int degree_of_freedom_row[DEG_FREEDOM] = {0, 1, 1, 1};   //right_left, up_down, diagonally up_down_left, diagonally up_down_right
int degree_of_freedom_col[DEG_FREEDOM] = {-1, 0, -1, 1}; //right_left, up_down, diagonally up_down_left, diagonally up_down_right

int main() 
{
  generate_matrix(false);
  int processor_count = omp_get_num_procs();
  cout << processor_count << " << processor count!" << endl;
  omp_set_num_threads(4);
  cout << omp_get_num_threads() << " << num threads " << endl;
  for (int t = 1; t <= processor_count; t++) {
    for (int k = 1; k < 20; k++) {
      int cnt = 0;
      double begin = omp_get_wtime();
      #pragma omp parallel for reduction (+:cnt)
      for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLUMNS; j++) {
            cnt += count_pal_all_dir(k, i, j);
        }
      }
      double end = omp_get_wtime();
      double time_spent = end - begin;
      cout <<fixed<< cnt << " palindromes of size " << k << " found in " << time_spent << " s. using " << omp_get_num_threads() << " Thread(s)" << endl;
    }
    cout << "********************************************************************************" << endl;
  }
}

void generate_matrix(bool print) {
  srand (time(NULL));
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
    if (check_pal_size_n_from_cell(pal_size, cell_row, cell_col, degree_of_freedom_row[i], degree_of_freedom_col[i]))
      cnt++;
  }

  return cnt;
}

bool check_pal_size_n_from_cell(int palindrome_size, int cell_row, int cell_col, int row_direction, int col_direction) {
  stack<char> st = {};
  int half_size;

  if ((palindrome_size&1) == 0)
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
      if ((palindrome_size&1) == 1)
        st.pop();
    }
    cell_row += row_direction;
    cell_col += col_direction;
  }

  return true;
}