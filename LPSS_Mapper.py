import sys
import ast
import networkx as nx


def main():
    for line in sys.stdin:
        key, value = ast.literal_eval(line.strip())
        status, label, adjacency_list = value

        if status:
            for neighbour in adjacency_list:
                print((neighbour,label))

        print((key,value))

    return





if __name__ == "__main__":
    main()
