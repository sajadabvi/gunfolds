# Open the file in read mode
with open("filtered_memoryusage.txt", "r") as file:
    content = file.read()

import re

# Define a regular expression pattern to match the desired phrases and the numbers
pattern = r"number of optimal solutions is (\d+)\\"

# Find all matches in the content
matches = re.findall(pattern, content)

# Convert the matched numbers to integers and store them in a list
numbers = [int(match) for match in matches]

# Print the list of numbers
print(numbers)
