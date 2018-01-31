import os
import json
def get_filepaths(directory):
    """
    :param directory: where file stores
    :return: return a list of all files in directory
    """
    file_paths = []
    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.
    return file_paths
print('haha')
new_file=open('./facebook_data.csv','w+')
files=get_filepaths('../facebook_data')
for file in files:
    f=open(file)
    content=f.read()
    content=json.loads(content)
    new_file.write(content['title'].replace(',',''))
    new_file.write(',')
    new_file.write(content['published'][:10])
    new_file.write('\n')
    f.close()
new_file.close()
