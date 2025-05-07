#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 6                  // Number of nodes in the graph
#define TOTAL_NODES N        // Total number of nodes in the graph

int visited[TOTAL_NODES];
int queue[TOTAL_NODES];
int front = 0, rear = 0;

// Custom graph adjacency list
int* graph[TOTAL_NODES];

// Function to add an edge to the graph
void add_edge(int from, int to) {
    // Allocate memory for the adjacency list if it's not already allocated
    static int initialized = 0;
    if (!initialized) {
        for (int i = 0; i < TOTAL_NODES; i++) {
            graph[i] = malloc(TOTAL_NODES * sizeof(int)); // Allocating space for up to N neighbors per node
        }
        initialized = 1;
    }

    // Simply add the node to the adjacency list (directed graph)
    graph[from][to] = 1;
    graph[to][from] = 1;  // Assuming an undirected graph, if directed, omit this line.
}

// Get neighbors of a node
void get_neighbors(int node, int* neighbors, int* count) {
    *count = 0;
    for (int i = 0; i < TOTAL_NODES; i++) {
        if (graph[node][i] == 1) {  // If there's an edge
            neighbors[(*count)++] = i;
        }
    }
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

            int neighbors[TOTAL_NODES], count;
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
    // Initialize the graph (custom graph with 6 nodes)
    add_edge(0, 1);  // Node 0 is connected to Node 1
    add_edge(0, 2);  // Node 0 is connected to Node 2
    add_edge(1, 3);  // Node 1 is connected to Node 3
    add_edge(2, 3);  // Node 2 is connected to Node 3
    add_edge(2, 4);  // Node 2 is connected to Node 4
    add_edge(3, 5);  // Node 3 is connected to Node 5
    add_edge(4, 5);  // Node 4 is connected to Node 5

    int start_node = 0; // Start from Node 0

    double start_time = omp_get_wtime(); // Start timer

    bfs_parallel(start_node);

    double end_time = omp_get_wtime();   // End timer

    printf("Visited nodes in BFS order:\n");
    for (int i = 0; i < TOTAL_NODES; i++) {
        if (visited[i]) {
            printf("Node %d ", i);
        }
    }
    printf("\n");

    printf("Visited nodes in BFS order:\n");

    int first = 1; // Flag to manage the space between nodes
    for (int i = 0; i < TOTAL_NODES; i++) {
        if (visited[i]) {
            if (!first) {
                printf(" ");
            }
            first = 0;

            // Output in the format (row, col)
            int row = i / N, col = i % N;
            printf("(%d, %d)", row, col);
        }
    }
    printf("\n");
    printf("Time taken for parallel BFS: %f seconds\n", end_time - start_time);

    return 0;
}
