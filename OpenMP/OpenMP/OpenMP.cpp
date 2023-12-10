#include <iostream>
#include <cstdlib>
#include <omp.h>

using namespace std;

void randomiseMatrix(int** matrix, int N) {

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			matrix[i][j] = rand() % 11;
		}
	}
	return;
}

int main(int argc, char** argv) {

	int N;
	int threads;

	printf("Enter matrix size (integer value): ");
	cin >> N;

	printf("Enter max number of threads (integer value): ");
	cin >> threads;

	int** matrix_one;
	int** matrix_two;
	int** matrix_res;

	matrix_one = (int**)malloc(sizeof(int*) * N);
	matrix_two = (int**)malloc(sizeof(int*) * N);
	matrix_res = (int**)malloc(sizeof(int*) * N);

	for (int i = 0; i < N; i++) {
		matrix_one[i] = (int*)malloc(sizeof(int) * N);
	}

	for (int i = 0; i < N; i++) {
		matrix_two[i] = (int*)malloc(sizeof(int) * N);
	}

	for (int i = 0; i < N; i++) {
		matrix_res[i] = (int*)malloc(sizeof(int) * N);
	}

	randomiseMatrix(matrix_one, N);
	randomiseMatrix(matrix_two, N);

	int i, j, k;
	double start, end;

	for (int thr = 1; thr <= threads; thr++) {

	omp_set_num_threads(thr);

	start = omp_get_wtime();

#pragma omp parallel for shared(matrix_one, matrix_two, matrix_res) private(i, j, k)
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			matrix_res[i][j] = 0;
			for (k = 0; k < N; k++) {
				matrix_res[i][j] += (matrix_one[i][k] * matrix_two[k][j]);
			}
		}
	}

	end = omp_get_wtime();

	printf("Number of seconds with %d threads: %f", thr, (end - start));
	printf("\n");
	}

	return 0;
}