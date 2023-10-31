import numpy as np
import multiprocessing

# Load balancing algorithm function
def load_balance(matrix_a, matrix_b, num_processes, process_id):
    num_rows_a, num_cols_a = matrix_a.shape
    num_rows_b, num_cols_b = matrix_b.shape

    # Calculate the number of rows each process should handle
    rows_per_process = num_rows_a // num_processes
    extra_rows = num_rows_a % num_processes

    # Calculate the start and end row indices for the current process
    start_row = process_id * rows_per_process
    end_row = start_row + rows_per_process

    # Adjust the end row for the last process to account for extra rows
    if process_id == num_processes - 1:
        end_row += extra_rows

    # Extract the corresponding block of rows from matrix A
    sub_matrix_a = matrix_a[start_row:end_row, :]

    # Perform matrix multiplication on the extracted block
    result = np.dot(sub_matrix_a, matrix_b)

    return process_id, result, start_row, end_row

# Function to perform matrix multiplication
def matrix_multiply(matrix_a, matrix_b):
    return np.dot(matrix_a, matrix_b)

# Function to execute matrix multiplication in parallel
def parallel_matrix_multiply(process_id, matrix_a, matrix_b, num_processes):
    # Load balance the work based on process_id using your load_balance function
    process_id, result, start_row, end_row = load_balance(matrix_a, matrix_b, num_processes, process_id)
    return process_id, result, start_row, end_row

if __name__ == '__main__':
    num_processes = int(input("Enter the number of processes: "))
    
    # Load matrices from 'input.txt'
    with open('input.txt', 'r') as file:
        lines = file.read().splitlines()

    matrices = []
    current_matrix = []
    for line in lines:
        if line.strip():  # Check if the line is not empty
            current_matrix.append(list(map(float, line.split(','))))
        else:
            matrices.append(np.array(current_matrix))
            current_matrix = []

    # Append the last matrix
    matrices.append(np.array(current_matrix))

    # Split the matrices into matrix_a and matrix_b
    matrix_a = matrices[0]
    matrix_b = matrices[1]

    print(matrix_b.shape)

    if num_processes == 1:
        # Perform sequential matrix multiplication
        print("Performing sequential matrix multiplication.")
        result = matrix_multiply(matrix_a, matrix_b)
    else:
        # Perform parallel matrix multiplication
        print(f"Performing parallel matrix multiplication with {num_processes} processes.")

        pool = multiprocessing.Pool(num_processes)
        results = []

        # Distribute work among processes
        for i in range(num_processes):
            result = pool.apply_async(parallel_matrix_multiply, args=(i, matrix_a, matrix_b, num_processes))
            results.append(result)

        # Collect and print intermediate results from processes
        combined_result = np.zeros((matrix_a.shape[0], matrix_b.shape[1]))
        for i, result in enumerate(results):
            process_id, partial_result, start_row, end_row = result.get()
            print(f"Process {process_id} finished with intermediate result:")
            print(partial_result)
            combined_result[start_row:end_row, :] = partial_result

        result = combined_result

    # Print the final result
    print("Final Result:")
    print(result)

    # Write the result to 'Output.txt'
    np.savetxt('Output.txt', result, fmt='%.2f', delimiter=', ')
