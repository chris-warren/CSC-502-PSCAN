
import sys
import ast
from typing import Dict, Iterable, List, Set, Tuple
from collections import defaultdict
from similarity_score.similarity import load_adjacency_list



def create_filtered_adjlist(EPSILON: float = 0.5) -> Dict[int, List[int]]:
   """
    Parses output from std::istream stream LPCC_Reducer, whose output is in the format of (edge, similarity_score).

    This function:
    - filters/remove out edges whose similarity_scores are lower than EPSILON
    

    Parameters
    ----------
    EPSILON : float
        epsilon constant for similarity_score.

    Returns
    -------
    Dict[int, List[int]]
        Mapping from node -> List of neighbors.

    Raises
    ------
    ValueError
        If input value is not in the format of (edges,similarity_score), then raises a valueerror.
    
    """

    return_val = defaultdict(list)

    for line in sys.stdin:

        edge, similarity_score = ast.literal_eval(line.strip())

        if not (isinstance(edges, Tuple) and isinstance(similarity_score,(int,float))):
            raise ValueError(f"Invalid input line: {line}")

        if similarity_score >= EPSILON:

            x, y = edge
            return_val[x].append(y)
            return_val[y].append(x)
    
    return return_val

            
def LPCC_emitter(filtered_adjlist : Dict[int, List[int]]) -> None: 
     """
    

    This function:
    - emits output for `LPSS_Mapper` in the form of (vertex_id, (True, vertex_id, vertex_id_adjlist))
    

    Parameters
    ----------
    filtered_adjlist : Dict[int, List[int]]
        filtered_adjlist whose edges have similarity score >= EPSILON

    Returns
    -------
    None

    Raises
    ------
    
    """

    
    for key, current_adjlist in filtered_adjlist.items():
        
        status = True
        label = key
        print(f"({key},({status},{label},{current_adjlist}))")






if __name__ == "__main__":

    # LFR_buffer_5000 = load_adjacency_list("data/adjlists/lfr_5000.adjlist")
    # LFR_buffer_10000 = load_adjacency_list("data/adjlists/lfr_10000.adjlist")

    # ba_buffer_100000 = load_adjacency_list("data/adjlists/ba_100000.adjlist")
    # ba_buffer_200000 = load_adjacency_list("data/adjlists/ba_200000.adjlist")

    adjlist = create_filtered_adjlist()
    LPCC_emitter(adjlist)


