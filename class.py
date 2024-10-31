import os

annotated_path = "F:\\2.detection-2ndData\\newdata_test\\annotated"

# Define the data to be changed
old_data = "1 "
new_data = "0 "

# Iterate through each .txt file in the directory
for txt_file in os.listdir(annotated_path):
    if txt_file.endswith(".txt"):
        file_path = os.path.join(annotated_path, txt_file)
        # Read the content of the file
        with open(file_path, "r") as file:
            lines = file.readlines()

        # Modify the line containing the data you want to change
        modified_lines = [line.replace(old_data, new_data) for line in lines]

        # Write the modified content back to the file
        with open(file_path, "w") as file:
            file.writelines(modified_lines)

print("Data in .txt files modified successfully.")
