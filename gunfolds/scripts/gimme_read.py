import os
import numpy as np
import csv


def read_csv_files(path):
    data = []
    files = sorted(os.listdir(path))  # Sort the files
    for filename in files:
        if filename.endswith('.csv'):
            file_path = os.path.join(path, filename)
            with open(file_path, 'r') as file:
                csv_reader = csv.reader(file)
                next(csv_reader)  # Skip header if exists
                rows = []
                for row in csv_reader:
                    rows.append(row[0:10])  # Extract columns 5 to 10
                mat = np.array(rows, dtype=np.float32)
                matrix1 = np.array(mat[:, 0:5])
                matrix2 = np.array(mat[:, 5:10])
                sum_matrix = matrix1 + matrix2
                binary_matrix = (sum_matrix != 0).astype(int)
                data.append(binary_matrix)
    return data


def main():
    path = "/Users/sajad/GSU Dropbox Dropbox/Mohammadsajad Abavisani/Mac/Documents/PhD/Research/code/GIMME/gimme-master/gimme-master/output4/sum"
    csv_data = read_csv_files(path)

    # Convert to NumPy array
    np_data = np.array(csv_data)
    print("Shape of NumPy array:", np_data.shape)


if __name__ == "__main__":
    main()
