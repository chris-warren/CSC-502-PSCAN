from pathlib import Path
import shutil

 

 
mapper = Path("LPSS_final_mapper_result")
if mapper.exists():
    shutil.rmtree(mapper)
mapper.mkdir(parents=True)


 
mapper_temp = Path("LPSS_temp_mapper_result")
if mapper_temp.exists():
    shutil.rmtree(mapper_temp)
mapper_temp.mkdir(parents=True)
 
 
def LPSS_Mapper(path, temp_output):
    path = Path(path)
    tsv_files = list(path.glob("*.tsv"))
 
    
 
    for f in tsv_files:
        print(f.stem)
        current = 0

        activated_content = []
        non_activated_content = []
 
        with open(f) as file:
            for line in file:
                key, status, label, *adjacency_list = line.strip().split(",")
                adjacency_list = adjacency_list[0].split(" ")
                print(adjacency_list)
                
 
                if status:
                    current += 1
                    for neighbour in adjacency_list:
                        activated_content.append((neighbour, status, label, ",".join(adjacency_list)))
                else:
                    non_activated_content.append((key, status, label, ",".join(adjacency_list)))

        
 
        with open(f"LPSS_final_mapper_result/{f.stem}.tsv", "a") as file:
            if len(non_activated_content):
                for element in non_activated_content:
                    file.write(f"{element[0]},{element[1]},{element[2]},{element[3]}\n")
 
        with open(f"{temp_output}/{f.stem}.tsv", "w") as file:
            if len(activated_content):
                for element in activated_content:
                    file.write(f"{element[0]},{element[1]},{element[2]},{element[3]}\n")
 
    return bool(current)
 
 
def LPSS_Reducer(path, temp_output):
    
    
    path = Path(path)
    tsv_files = list(path.glob("*.tsv"))

    for f in tsv_files:
        print(f.stem)

        current_dict = {}
        with open(f) as file:

            for line in file:
                vertex_id, status, label, *adjacency_list = line
                current_dict


        
    

    return

 
 
if __name__ == "__main__":
    ret_val = LPSS_Mapper("parsed_input/", "LPSS_temp_mapper_result/")

    if not ret_val:
        print("LPSS procedure complete!")
 
    LPSS_Reducer("LPSS_temp_mapper_result","LPSS_temp_mapper_result")
    
    while True:
        ret_val = LPSS_Mapper("")
        break