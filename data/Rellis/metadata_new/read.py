import os
import pickle

# Get all pkl files in the current directory
pkl_files = [f for f in os.listdir('.') if f.endswith('.pkl')]

# Traverse each pkl file
for pkl_file in pkl_files:
    # Open the pkl file and load the data
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    # Convert data to string form
    data_str = str(data)

    # Create the corresponding txt file name
    txt_file = pkl_file.replace('.pkl', '.txt')

    # Write the string to a txt file
    with open(txt_file, 'w') as f:
        f.write(data_str)

    print(f"The data has been successfully saved to the {txt_file} file.")
