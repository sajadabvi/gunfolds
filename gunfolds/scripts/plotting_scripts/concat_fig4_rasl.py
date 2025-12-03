import os
import glob
from gunfolds.utils import zkl  # Assuming you have this module available


def read_and_structure_zkl_files(directory):
    # Initialize the result as a 3x3 structure of empty lists
    result = [[[] for _ in range(3)] for _ in range(3)]

    # Loop over all .zkl files in the directory
    for filepath in glob.glob(os.path.join(directory, '*.zkl')):
        # Load the data from the file
        x = zkl.load(filepath)  # Assuming x is a 3x3 structure

        # Distribute the values into the result structure
        for i in range(3):  # Iterate over rows
            for j in range(3):  # Iterate over columns
                result[i][j].append(x[i][j])

    return result


if __name__ == "__main__":
    # Example usage
    directory_path = "/Users/mabavisani/code_local/mygit/gunfolds/gunfolds/scripts/test_rasl_fig4"
    result = read_and_structure_zkl_files(directory_path)
    # 'result' is now a 3x3 structure where each sublist contains data
    # from all files.
    print(result)