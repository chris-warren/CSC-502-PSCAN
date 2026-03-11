import sys
import ast


def main():

    current_min = min([int(ast.literal_eval(line.strip())[1]) for idx, line in enumerate(list(sys.stdin)) if idx%2])


    for line1, line2 in zip(sys.stdin,sys.stdin):
        key, value = ast.literal_eval(line2.strip())
        status, label, adjacency_list = value

        if current_min < label:
            status = True
            label = current_min
        else:
            status = False
        
        print((key,(status,label,adjacency_list)))
    return
  
  
if __name__ == "__main__":
    main()