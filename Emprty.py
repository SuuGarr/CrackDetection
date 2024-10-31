import os

# Set the directory where your .txt files are stored
directory = 'F:\\2.detection-2ndData\\newdata_test\\annotated'

# Loop through each file in the directory
for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        # Construct the full path to the file
        filepath = os.path.join(directory, filename)
        # Read the content of the file
        with open(filepath, 'r') as file:
            content = file.read().strip()
        # Check if the content matches '0 0 0 0 0'
        if content == '0 0 0 0 0':
            # Open the file in write mode and empty it
            with open(filepath, 'w') as file:
                file.write('')

print("Files have been processed.")
