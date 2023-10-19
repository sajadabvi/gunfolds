# Initialize an empty list to store the numbers
numbers_list = []

# Open the file in read mode
with open("filtered_memoryusage.txt", "r") as file:
    for line in file:
        # Check if the line contains "batch" and "K  COMPLETED"
        if "batch" in line and "K  COMPLETED" in line:
            # Extract the number between "batch" and "K  COMPLETED"
            start_index = line.find("batch") + len("batch")
            end_index = line.find("K  COMPLETED")
            number = line[start_index:end_index].strip()

            # Check if the extracted content is a number
            if number.isdigit():
                numbers_list.append(int(number))

# Print the list of numbers
print(numbers_list)
