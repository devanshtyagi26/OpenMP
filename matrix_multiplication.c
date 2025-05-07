#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 500      // Matrix size
#define BS 50      // Block size

int main() {
    int A[N][N], B[N][N], C[N][N];
    int i, j, k;

    // Initialize matrices
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++) {
            A[i][j] = i + j;
            B[i][j] = i - j;
            C[i][j] = 0;
        }

    double start = omp_get_wtime();

    #pragma omp parallel for collapse(2) private(i,j,k)
    for (int ii = 0; ii < N; ii += BS) {
        for (int jj = 0; jj < N; jj += BS) {
            for (int kk = 0; kk < N; kk += BS) {
                for (i = ii; i < ii + BS && i < N; i++) {
                    for (j = jj; j < jj + BS && j < N; j++) {
                        for (k = kk; k < kk + BS && k < N; k++) {
                            C[i][j] += A[i][k] * B[k][j];
                        }
                    }
                }
            }
        }
    }

    double end = omp_get_wtime();
    printf("Time taken with blocking: %f seconds\n", end - start);
    return 0;
}

// gcc -fopenmp matrix_multiplication.c -o matrix_multiplication