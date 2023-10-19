# Initialize an empty set to store the numbers
number_set = set()

# Open the file in read mode
with open("jobIDs.txt", "r") as file:
    for line in file:
        # Find the position of "out" and "-"
        out_index = line.find("out")
        dash_index = line.find("-")

        # Check if both "out" and "-" are present in the line
        if out_index != -1 and dash_index != -1:
            # Extract the number between "out" and "-"
            number = line[out_index + 3:dash_index].strip()

            # Check if the extracted content is a number
            if number.isdigit():
                number_set.add(int(number))

# Print the set of numbers
print(number_set)

formatted_string = ""

# Iterate through the number_set and format each number
for number in number_set:
    formatted_string += f'"{number}" '

# Remove the trailing space
formatted_string = formatted_string.strip()

# Print the formatted string
print(formatted_string)