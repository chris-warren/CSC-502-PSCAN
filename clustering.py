import networkx as nx
import json

# Config
EPSILON = 1.5
G_lfr_5000_adjlist = nx.read_adjlist('data/adjlists/lfr_5000.adjlist')


def main():
    

    input_example = [(1,('activated',1,[2,3])),(2,('activated',2,[1])),(3,('activated',3,[1]))];
    current_program = clustering(input_example);
    
    
    current_program.LPCC_MAPPER()
    return





class clustering:

    s = 1
    def __init__(self,input_list):
        """
        class constructor
        input: [(key,value),(key,value),...]
        
        value == (status, label, adjacency_list)
        """
        self.input_list = input_list
    
    def LPCC_MAPPER(self):
        """
        mapper function

        return [(key,value),(key,value),...]

        value == (key, label) or value == (key, (status, label, adjacency_list))
        
        """

        output = []
        for _ in self.input_list:
            key, value = _
            status, label, adjacency_list = value

            if (status):
                for adj_list in adjacency_list:
                    temp_key = value
                    output.append((temp_key,label))
        
            output.append((key,value))
        
        return output

    def LPCC_REDUCER(self,input_list):
        """
        input: LPCC_Mapper() return value


        output: []
        
        
        """

        for _ in input_list:
            key, value = _
            status, label, adjacency_list = value
            temp = 0

            cur_min = INT_MAX

            for value in adjacency_list:
                if len(value)>1:
                    temp = value
                else:
                    for adj_list in adjacency_list:
                        cur_min = min(cur_min,adj_list)
            if label > cur_min:
                label = cur_min
                status = True
            else:
                status = False






if __name__ == "__main__":
    main()

