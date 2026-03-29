from LPSS import *
from LPSS_pyspark import *
import csv


def cluster_results(sim_path, epsilon, PROJECT_ROOT, sc):
    """
    Wrapper function for LPSS and LPSS_pyspark
    writes resulting clusters to disc
    """
    parsed_input_path = create_filtered_adjlist_and_LPCC_emitter(
        tsv_file=sim_path, EPSILON=epsilon, project_root=PROJECT_ROOT
    )
    result = run_lpcc(
        parsed_input_path,
        project_root=PROJECT_ROOT,
        sc=sc,
    )  # file is comma-separated despite .tsv extension

    components = (
        result.map(lambda line: (line.split(",")[2], line.split(",")[0]))
        .groupByKey()
        .mapValues(sorted)
        .collect()
    )
    with open(
        f'{PROJECT_ROOT}/data/output/clusters/{str(sim_path).split("/")[-1].split(".")[0]}e{str(epsilon).replace(".", "")}.csv',
        "w",
        newline="",
    ) as csvfile:
        writer = csv.writer(csvfile)
        for row in sorted(components):
            writer.writerow(row)

    return components
