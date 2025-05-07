#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 4                  // Grid size (N x N)
#define TOTAL_NODES (N * N)  // Total number of nodes in the graph

int visited[TOTAL_NODES];
int queue[TOTAL_NODES];
int front = 0, rear = 0;

// Get node index in 1D array from 2D coordinates
int toIndex(int row, int col) {
    return row * N + col;
}

// Add neighbors of a node in a grid (up, down, left, right)
void get_neighbors(int node, int* neighbors, int* count) {
    int row = node / N;
    int col = node % N;
    *count = 0;

    if (row > 0) neighbors[(*count)++] = toIndex(row - 1, col);     // Up
    if (row < N - 1) neighbors[(*count)++] = toIndex(row + 1, col); // Down
    if (col > 0) neighbors[(*count)++] = toIndex(row, col - 1);     // Left
    if (col < N - 1) neighbors[(*count)++] = toIndex(row, col + 1); // Right
}

void bfs_parallel(int start) {
    // Parallel initialization of visited array
    #pragma omp parallel for
    for (int i = 0; i < TOTAL_NODES; i++) {
        visited[i] = 0;
    }

    queue[rear++] = start;
    visited[start] = 1;

    while (front != rear) {
        int level_size = rear - front;

        #pragma omp parallel for
        for (int i = 0; i < level_size; i++) {
            int current = queue[front + i];

            int neighbors[4], count;
            get_neighbors(current, neighbors, &count);

            for (int j = 0; j < count; j++) {
                int neighbor = neighbors[j];

                if (!visited[neighbor]) {
                    // Mark visited and enqueue
                    visited[neighbor] = 1;
                    #pragma omp critical
                    {
                        queue[rear++] = neighbor;
                    }
                }
            }
        }

        front += level_size;
    }
}

int main() {
    int start_node = 0; // Start from top-left of the grid

    double start_time = omp_get_wtime(); // Start timer

    bfs_parallel(start_node);

    double end_time = omp_get_wtime();   // End timer

    printf("Visited nodes in BFS order:\n");
    for (int i = 0; i < TOTAL_NODES; i++) {
        if (visited[i]) {
            int row = i / N, col = i % N;
            printf("(%d, %d) ", row, col);
        }
    }
    printf("\n");

    printf("Time taken for parallel BFS: %f seconds\n", end_time - start_time);

    return 0;
}