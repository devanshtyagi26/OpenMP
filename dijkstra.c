#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <omp.h>

#define N 6            // Number of nodes in the graph
#define INF INT_MAX    // Representation of infinity (no path)

int graph[N][N];      // Adjacency matrix to represent the graph
int dist[N];          // Array to store shortest distances from the source
int visited[N];       // Array to track visited nodes

// Function to initialize the graph
void init_graph() {
    // Example graph (Adjacency matrix representation)
    int g[N][N] = {
        {0, 7, INF, INF, INF, 2},
        {7, 0, 8, INF, INF, 3},
        {INF, 8, 0, 6, INF, INF},
        {INF, INF, 6, 0, 1, 9},
        {INF, INF, INF, 1, 0, 4},
        {2, 3, INF, 9, 4, 0}
    };
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            graph[i][j] = g[i][j];
        }
    }
}

// Function to implement Dijkstra's algorithm in parallel
void dijkstra_parallel(int source) {
    // Initialize distance array and visited array
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        dist[i] = INF;      // Initially set all distances to infinity
        visited[i] = 0;     // Mark all nodes as unvisited
    }
    dist[source] = 0; // Distance to the source node is 0

    // Dijkstra's main loop
    for (int count = 0; count < N; count++) {
        // Find the unvisited node with the smallest tentative distance (sequential)
        int u = -1;
        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            if (!visited[i] && (u == -1 || dist[i] < dist[u])) {
                #pragma omp critical
                {
                    if (!visited[i] && (u == -1 || dist[i] < dist[u])) {
                        u = i;
                    }
                }
            }
        }

        // Mark the node as visited
        visited[u] = 1;

        // Update distances for neighbors of u (parallel relaxation)
        #pragma omp parallel for
        for (int v = 0; v < N; v++) {
            if (graph[u][v] != INF && !visited[v]) {
                int new_dist = dist[u] + graph[u][v];
                if (new_dist < dist[v]) {
                    dist[v] = new_dist;
                }
            }
        }
    }
}

// Function to print the shortest distances
void print_distances() {
    printf("Shortest distances from the source node:\n");
    for (int i = 0; i < N; i++) {
        printf("Node %d: %d\n", i, dist[i]);
    }
}

int main() {
    int source = 0; // Starting node (Node 0)

    init_graph();   // Initialize the graph

    double start_time = omp_get_wtime(); // Start timer

    dijkstra_parallel(source); // Run the parallel Dijkstra's algorithm

    double end_time = omp_get_wtime();   // End timer

    print_distances(); // Print the results

    printf("Time taken for parallel Dijkstra: %f seconds\n", end_time - start_time);

    return 0;
}
