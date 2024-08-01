import os
import shutil

# Define the base directory
base_dir = '.'  # Change this to your base directory if necessary

# Walk through the directory structure
for root, dirs, files in os.walk(base_dir):
    if 'individual' in dirs:
        individual_path = os.path.join(root, 'individual')
        std_errors_path = os.path.join(individual_path, 'StdErrors')
        
        # Create StdErrors directory if it doesn't exist
        if not os.path.exists(std_errors_path):
            os.makedirs(std_errors_path)
        
        # Move all *StdErrors.csv files to the StdErrors directory
        for file_name in os.listdir(individual_path):
            if file_name.endswith('StdErrors.csv'):
                src_file = os.path.join(individual_path, file_name)
                dest_file = os.path.join(std_errors_path, file_name)
                shutil.move(src_file, dest_file)

print("Operation completed successfully.")

