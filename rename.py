import os

# Path to the directory containing the .bmp files
directory = "F:\\2.detection-2ndData\\newdata_test\\annotated"

# Retrieve a list of .bmp files in the specified directory
files = [file for file in os.listdir(directory) if file.endswith('.txt')]

# Sort files if needed; depends on original naming convention
files.sort()

# Rename files
for count, filename in enumerate(files, start=1):
    new_filename = f"{count}.txt"
    original_path = os.path.join(directory, filename)
    new_path = os.path.join(directory, new_filename)
    os.rename(original_path, new_path)
    print(f"Renamed {filename} to {new_filename}")
