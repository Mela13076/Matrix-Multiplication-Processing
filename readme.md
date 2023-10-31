# Matrix Multiplication Code

This repository contains code for performing matrix multiplication, both sequentially and in parallel, using Python. You can choose to run either the `matrixmul.py` or `matmulperform.py` file to perform matrix multiplication tasks.

## Prerequisites

Before running the code, you need to ensure that you have the necessary libraries installed. You can do this using `pip3`:

```bash
pip3 install numpy multiprocessing time csv
```


## Running the Code

Follow these steps to run the code:

1. Open the code folder in your preferred code editor (e.g., Visual Studio Code).
2. Make sure that all the files within the code folder are in the same directory.
3. Open a terminal in your code editor or use your system's terminal.
4. To perform sequential matrix multiplication, run the following command:

```bash
python3 matrixmul.py
```
You will be prompted to enter the number of processes (usually 1 for sequential execution).

5. To  perform parallel matrix multiplication and compare performance, run the following command:

```bash
python3 matmulperform.py
```
You will again be prompted to enter the number of processes (e.g., 2, 4, or more).

6. follow the prompts in the terminal to proceed with the matrix multiplication tasks.


## File Descriptions

- matrixmul.py: Contains functions for matrix multiplication, load balancing, and parallel matrix multiplication.

- matmulperform.py: Executes matrix multiplication tasks sequentially and in parallel. It measures and compares execution times for different matrix sizes and processes.

## Notes

- Running multiple processes simultaneously can consume substantial CPU and memory resources. Monitor your system's resource usage if you experience performance issues.

- The code generates CSV files (Seq_exe.csv and Paral_exe.csv) containing execution time data for analysis.
