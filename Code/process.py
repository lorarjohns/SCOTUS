import json
import os
import re 
import glob
import pandas as pd
from tqdm import tqdm

cwd = os.getcwd() + '/'

def rename_to_text():
    new_names = []
    old_names = glob.glob('*.json')

    for name in old_names:
            new_name = name.split('.')[0] + '.txt'   
            os.rename(os.path.join(cwd, name), os.path.join(cwd, new_name))
            new_names.append(new_name)

    return new_names     

def validate_json(files): 
    validated = []  
    for index, name in tqdm(enumerate(files)):
        filepath = os.path.join(cwd,name)
        with open(filepath,'r+') as f:
            data = f.read()            
            converted = re.sub("(\w+):(.+)", r'"\1":\2', data)
            f.seek(0)
            f.write(converted)
            f.truncate()
    #rename            
        new_name = name[:-4] + '.json'  
        os.rename(os.path.join(cwd, name), os.path.join(cwd, new_name))
        validated.append(new_name)        
    print(f"All corrupt files converted to valid json.")        
    return validated

if __name__ == "__main__":	
	rename_to_text()
	validate_json(glob.glob('*.json'))
