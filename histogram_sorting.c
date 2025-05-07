#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 100000    // Number of elements
#define MAX_VAL 100 // Max possible value (range: 0 to MAX_VAL)
#define NUM_BUCKETS 10 // Number of buckets for histogram

// Comparison function for sorting integers
int compare(const void *a, const void *b) {
    return (*(int *)a - *(int *)b);
}

int main() {
    int data[N], histogram[MAX_VAL] = {0};
    int *buckets[NUM_BUCKETS];
    int bucket_counts[NUM_BUCKETS] = {0};

    // Step 1: Parallel random data generation using private random generators for each thread
    #pragma omp parallel
    {
        unsigned int seed = omp_get_thread_num();  // Unique seed for each thread

        #pragma omp for
        for (int i = 0; i < N; i++) {
            data[i] = rand_r(&seed) % MAX_VAL;  // Using rand_r for thread-safe random number generation
        }
    }

    double start = omp_get_wtime();

    // Step 2: Parallel histogram calculation
    #pragma omp parallel
    {
        int local_histogram[MAX_VAL] = {0};

        #pragma omp for
        for (int i = 0; i < N; i++) {
            local_histogram[data[i]]++;
        }

        #pragma omp critical
        for (int i = 0; i < MAX_VAL; i++) {
            histogram[i] += local_histogram[i];
        }
    }

    // Step 3: Distribute elements into buckets
    int bucket_size = (MAX_VAL + NUM_BUCKETS - 1) / NUM_BUCKETS;
    for (int i = 0; i < NUM_BUCKETS; i++) {
        buckets[i] = malloc(N * sizeof(int)); // Allocate memory for each bucket
        bucket_counts[i] = 0; // Initialize bucket counts
    }

    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < N; i++) {
            int b = data[i] / bucket_size;
            if (b >= NUM_BUCKETS) b = NUM_BUCKETS - 1;

            int index;
            #pragma omp atomic capture
            index = bucket_counts[b]++;

            buckets[b][index] = data[i];
        }
    }

    // Step 4: Sort each bucket in parallel
    #pragma omp parallel for
    for (int i = 0; i < NUM_BUCKETS; i++) {
        qsort(buckets[i], bucket_counts[i], sizeof(int), compare);
    }

    // Step 5: Merge sorted buckets into the final sorted array
    int index = 0;
    for (int i = 0; i < NUM_BUCKETS; i++) {
        for (int j = 0; j < bucket_counts[i]; j++) {
            data[index++] = buckets[i][j];
        }
        free(buckets[i]); // Free memory after merging
    }

    double end = omp_get_wtime();
    printf("Distributed histogram sort completed in %f seconds.\n", end - start);

    // Optional: Print some sorted data (for verification)
    printf("Sorted Histogram:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", data[i]);
    }
    printf("\n");
    return 0;
}
