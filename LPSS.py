
from pathlib import Path
import sys
import ast
from typing import Dict, Iterable, List, Set, Tuple
from collections import defaultdict



def create_filtered_adjlist_and_LPCC_emitter(EPSILON: float = 0.5) -> None:
    
    
    tsv_files = list(Path("./similarity/data/data/adjlists/").glob("*.tsv"))

    for f in tsv_files:
        print(f)

    for f in tsv_files:
        current_dict = defaultdict(list)
        with open(f) as file:
            next(file)
            for line in file:
                current_row = line.strip().split('\t')
                u,v, current_similarity = current_row
                u = u
                v = v
                if(float(current_similarity)>= EPSILON):
                    current_dict[u].append(v)
                    current_dict[v].append(u)
                        
                 
        
        with open(f'filtered_adjlists/filtered_edge_{f.stem}.tsv',"w") as file:
             
             for key in current_dict:
                  neighbours = " ".join(map(str,current_dict[key]))
                  file.write(f"{key}\t{neighbours}\n")
             
            
    
        with open(f'parsed_input/parse_{f.stem}.tsv',"w") as file:

            
            for key, current_adjlists in current_dict.items():
                  current_adjlists = " ".join(current_adjlists)
                  
                  file.write(f'{key},{True},{key},{current_adjlists}\n')
    
    return

            


if __name__ == "__main__":
    create_filtered_adjlist_and_LPCC_emitter()


