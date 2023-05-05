#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <queue>
#include <stack>
#include <string>
#include <thread>
#include <vector>

using namespace std;

#define ROWS 100
#define COLUMNS 100
#define DEG_FREEDOM 4
#define SEARCH_MAX_SIZE 2
#define MAX_ALLOWED_PROCESSORS 4

struct Answer {
  int palindrome_count;
  int palindrome_size;
  double time_spent;
  int th_num;
} answer;

struct Compare {
  bool operator()(const Answer &lhs, const Answer &rhs) {
    if (lhs.th_num == rhs.th_num)
      return lhs.palindrome_size > rhs.palindrome_size;
    return lhs.th_num > rhs.th_num;
  }
};

int *create_search_size_array_and_fill(int search_max_size, int world_size);

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
  MPI_Init(NULL, NULL);
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Create Answer datatype
  const int nitems = 4;
  int blocklengths[4] = {1, 1, 1, 1};
  MPI_Datatype types[4] = {MPI_INT, MPI_INT, MPI_DOUBLE, MPI_INT};
  MPI_Datatype mpi_answer_type;
  MPI_Aint offsets[4];

  offsets[0] = offsetof(Answer, palindrome_count);
  offsets[1] = offsetof(Answer, palindrome_size);
  offsets[2] = offsetof(Answer, time_spent);
  offsets[3] = offsetof(Answer, th_num);
  MPI_Type_create_struct(nitems, blocklengths, offsets, types,
                         &mpi_answer_type);
  MPI_Type_commit(&mpi_answer_type);

  generate_matrix(false);

  int processor_count = omp_get_num_procs();
  // Because we are allocating buffer for gathering palindromes answers we
  // should set a threshold so all computers limited to use only
  // MAX_ALLOWED_PROCESSORS to not oveflow buffer
  if (processor_count > MAX_ALLOWED_PROCESSORS)
    processor_count = MAX_ALLOWED_PROCESSORS;

  omp_set_num_threads(processor_count);
  priority_queue<struct Answer, std::vector<Answer>, Compare> anspq = {};

  int *search_size_array = NULL;
  int *sub_search_size_array = (int *)calloc(SEARCH_MAX_SIZE, sizeof(int));

  if (world_rank == 0) {
    search_size_array =
        create_search_size_array_and_fill(SEARCH_MAX_SIZE, world_size);
  }

  MPI_Scatter(search_size_array, SEARCH_MAX_SIZE, MPI_INT,
              sub_search_size_array, SEARCH_MAX_SIZE, MPI_INT, 0,
              MPI_COMM_WORLD);

  int send_answer_array_size = MAX_ALLOWED_PROCESSORS * SEARCH_MAX_SIZE;
  Answer *send_answer =
      (Answer *)calloc(send_answer_array_size, sizeof(Answer));
  for (int i = 0; i < send_answer_array_size; i++) {
    send_answer[i].th_num = {0};
  }

  int ans_counter = 0;
  for (int t = 0; t < processor_count; t++) {
    for (int k = 0; k < SEARCH_MAX_SIZE; k++) {
      if (sub_search_size_array[k] == -1)
        continue;

      send_answer[ans_counter].palindrome_size = sub_search_size_array[k];
      int cnt = 0;
      double begin = omp_get_wtime();
#pragma omp parallel for num_threads(t + 1) reduction(+ : cnt)
      for (int i = 0; i < ROWS; i++) {
        send_answer[ans_counter].th_num = omp_get_num_threads();
        for (int j = 0; j < COLUMNS; j++) {
          cnt +=
              count_pal_all_dir(send_answer[ans_counter].palindrome_size, i, j);
        }
      }
      double end = omp_get_wtime();
      send_answer[ans_counter].time_spent = end - begin;
      send_answer[ans_counter].palindrome_count = cnt;
      ans_counter++;
    }
  }

  int recv_answer_array_size =
      MAX_ALLOWED_PROCESSORS * SEARCH_MAX_SIZE * world_size;
  Answer *recv_answer = NULL;
  if (world_rank == 0) {
    recv_answer = (Answer *)calloc(recv_answer_array_size, sizeof(Answer));
  }

  MPI_Gather(&send_answer, send_answer_array_size, mpi_answer_type, recv_answer,
             send_answer_array_size, mpi_answer_type, 0, MPI_COMM_WORLD);

  if (world_rank == 0) {
    for (int i = 0; i < recv_answer_array_size; i++) {
      cout << "Palindrome Size: " << recv_answer[i].palindrome_size 
      << "  Palindrome Count:" << recv_answer[i].palindrome_count
      << "  thnum:" << recv_answer[i].th_num
      << "  time_spend:" << recv_answer[i].time_spent << endl << endl;

      if (recv_answer[i].th_num != 0) {
        anspq.push(recv_answer[i]);
      }
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

  free(search_size_array);
  free(sub_search_size_array);
  free(send_answer);
  free(recv_answer);

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
}

int *create_search_size_array_and_fill(int search_max_size, int world_size) {
  int *gather_array;
  int gather_array_size = search_max_size * world_size;
  gather_array = (int *)calloc(gather_array_size, sizeof(int));

  int each_world_share = search_max_size / world_size;

  int cnt = 1;
  for (int i = 0; i < world_size; i++) {
    for (int j = 0; j < search_max_size; j++) {
      if (j >= each_world_share || cnt > search_max_size) {
        gather_array[(i * search_max_size) + j] = -1;
        continue;
      }
      gather_array[(i * search_max_size) + j] = cnt;
      cnt++;
    }
  }

  for (int i = 0; i < gather_array_size; i++) {
    if (cnt > search_max_size)
      break;
    if (gather_array[i] == -1) {
      gather_array[i] = cnt;
      cnt++;
    }
  }

  cout << "Gather Array: number of worlds: " << world_size << endl;
  for (int i = 0; i < gather_array_size; i++)
    cout << gather_array[i] << endl;

  return gather_array;
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