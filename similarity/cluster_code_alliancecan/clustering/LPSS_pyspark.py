# SparkContext.getOrCreate().stop() for google colab so it's optional i guess for running on local machine
# sc = SparkContext("local", "LPCC")


def mapper(line):
    parts = line.strip().split(",")
    vertex_id, status, label = parts[0], parts[1], parts[2]
    adjacency_list = parts[3].split(" ") if len(parts) > 3 and parts[3] else []

    out = []
    out.append(
        (vertex_id, ("struct", status, label, parts[3] if len(parts) > 3 else ""))
    )

    if status == "True":
        for neighbor in adjacency_list:
            out.append((neighbor, ("label", label)))

    return out


def reducer(vertex_id, values):
    values = list(values)

    struct = next(
        (v for v in values if v[0] == "struct"), None
    )  # get rid of first line
    neighbor_labels = [int(v[1]) for v in values if v[0] == "label"]

    if struct is None:
        return None

    _, status, label, adjlist = struct
    min_neighbor = min(neighbor_labels, default=int(label))

    if min_neighbor < int(label):
        return f"{vertex_id},{True},{min_neighbor},{adjlist}"
    else:
        return f"{vertex_id},{False},{label},{adjlist}"


def run_lpcc(input_path, project_root, sc):
    rdd = sc.textFile(input_path)

    iteration = 0
    while True:
        print(f"--- Iteration {iteration} ---")

        mapped = rdd.flatMap(mapper)
        reduced = (
            mapped.groupByKey()
            .map(lambda x: reducer(x[0], x[1]))
            .filter(lambda x: x is not None)
        )

        active_count = reduced.filter(lambda line: line.split(",")[1] == "True").count()

        print(f"active vertices: {active_count}")
        rdd = reduced
        iteration += 1

        if active_count == 0:
            print("finish lpcc")
            break

    return rdd


# if __name__ == "__main__":

#     result = run_lpcc(
#         "parse_lfr_500.sim.tsv"
#     )  # file is comma-separated despite .tsv extension

#     print("\nFinal connected components:")
#     components = (
#         result.map(lambda line: (line.split(",")[2], line.split(",")[0]))
#         .groupByKey()
#         .mapValues(sorted)
#         .collect()
#     )

#     for label, members in sorted(components):
#         print(f"Component {label}: {members}")
