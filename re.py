import pandas as pd
import json

# Load the mapping dictionary
map_file = 'map.json'
map_dict = json.load(open(map_file))

# Define file paths
input_files = [
    # '/home/hieu/Work/new_casacom/path1.csv',
    # '/home/hieu/Work/new_casacom/result/home.csv',
    # '/home/hieu/Work/new_casacom/result/work.csv',
    # '/home/hieu/Work/new_casacom/result/leisure.csv',
    '/home/hieu/Work/new_casacom/result/path_compressed.csv'
]

output_files = [
    # 'path2.csv',
    # '/home/hieu/Work/new_casacom/result/home2.csv',
    # '/home/hieu/Work/new_casacom/result/work2.csv',
    # '/home/hieu/Work/new_casacom/result/leisure2.csv',
    '/home/hieu/Work/new_casacom/result/path_compressed2.csv'
]

# Process each file
for input_file, output_file in zip(input_files, output_files):
    print(f"Processing {input_file} to {output_file}")
    first_chunk = True
    
    for i, chunk in enumerate(pd.read_csv(input_file, chunksize=1000000)):
        print(f'Processing chunk {i}')
        # Apply mapping and drop rows where mapping results in NaN
        chunk = chunk[chunk['maid'].map(map_dict).notna()]
        chunk['maid'] = chunk['maid'].map(map_dict)
        chunk.to_csv(output_file, mode='a', index=False, header=first_chunk)
        first_chunk = False
    
    print(f"Done! Check {output_file}")
