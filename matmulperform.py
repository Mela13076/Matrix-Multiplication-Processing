import numpy as np
import multiprocessing
from matrixmul import matrix_multiply, parallel_matrix_multiply
import time
import csv

# Function to generate random 'n x n' matrices
def generate_random_matrix(n):
    return np.random.rand(n, n)

# Function to measure execution time for matrix multiplication
def measure_execution_time(matrix_a, matrix_b, num_processes):
    start_time = time.time()
    
    if num_processes == 1:
        # Perform sequential matrix multiplication
        matrix_multiply(matrix_a, matrix_b)
    else:
        # Perform parallel matrix multiplication
        pool = multiprocessing.Pool(num_processes)
        results = []

        # Distribute work among processes
        for i in range(num_processes):
            result = pool.apply_async(parallel_matrix_multiply, args=(i, matrix_a, matrix_b, num_processes))
            results.append(result)

        # Collect results from processes
        pool.close()
        pool.join()

    end_time = time.time()
    return end_time - start_time

if __name__ == '__main__':
    num_iterations = 100
    max_matrix_size = 50
    num_processes = int(input("Enter the number of processes: "))

    # Create CSV files to save execution times
    with open('Seq_exe.csv', 'w', newline='') as seq_file, open('Paral_exe.csv', 'w', newline='') as paral_file:
        seq_writer = csv.writer(seq_file)
        paral_writer = csv.writer(paral_file)

        for n in range(2, max_matrix_size + 1):
            seq_times = []
            paral_times = []

            for _ in range(num_iterations):
                matrix_a = generate_random_matrix(n)
                matrix_b = generate_random_matrix(n)

                seq_time = measure_execution_time(matrix_a, matrix_b, 1)
                paral_time = measure_execution_time(matrix_a, matrix_b, num_processes)

                seq_times.append(seq_time)
                paral_times.append(paral_time)

            # Calculate the average execution time
            avg_seq_time = sum(seq_times) / num_iterations
            avg_paral_time = sum(paral_times) / num_iterations

            # Write average execution times to CSV files
            seq_writer.writerow([n, avg_seq_time])
            paral_writer.writerow([n, avg_paral_time])
