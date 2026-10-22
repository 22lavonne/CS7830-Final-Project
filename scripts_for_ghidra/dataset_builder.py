import csv
import re
import sys
from pathlib import Path
from rdflib import Graph

# hard coded paths for the ttl file, its name, where the sparql queries are located, and where the csv file will be put
TTL_DIR = Path.home() / ".local" / "share" / "GhidrAssist" / "semgraph"
TTL_NAME = "combined-output.ttl"
QUERY_FILE = Path("queries") / "queries.sparql"
OUT_CSV = Path("dataset.csv")

# the prefixes for the sparql queries 
# where the prefixes specify the ontology for what the objects we are looking for
PREFIXES = {
    "": "http://www.semanticweb.org/jaspe/ontologies/2026/0/combined-ontology/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
}

# loading the queries in from the query file
# using regex to find the "# name: var", which specifies the start of a new query
pieces = re.split(r"^# name:\s*(\w+)\s*$", QUERY_FILE.read_text(), flags=re.MULTILINE)
queries = dict(zip(pieces[1::2], pieces[2::2]))   # names are at odd positions, query text right after

# function to run a given query, with inputs of the current knowledge graph and the current query
# each query should only return 1 row and 1 column, so return whatever the query returns
# or if nothing returned from the query, just return 0
def run_query(graph, query):
    rows = list(graph.query(query, initNs=PREFIXES))
    if not rows or rows[0][0] is None:
        return 0
    return rows[0][0].toPython()

# this function parses the program name to get features out of it
# the features gotten include the optimization level, if the file is obfuscated (0 for no, 1 for yes), 
# and if the executable is stripped (0 if no, 1 if yes)
def filename_features(name):
    """Pull info out of e.g. x86__64__lsb__unix-system-v__gcc-5.5.0__O0__no-obf__unstripped__coreutils-8.30__nice"""
    parts = name.split("__")
    opt = parts[5]           # e.g. "O0"
    obf = parts[6]           # "no-obf" or something else
    stripped = parts[7]      # "unstripped" or "stripped"
    return {
        "opt_level": int(opt[1]) if opt[1:].isdigit() else None,   # O0 -> 0, O3 -> 3, Os -> blank
        "is_obfuscated": 0 if obf == "no-obf" else 1,
        # "is_stripped": 0 if stripped == "unstripped" else 1,
        # "meta_arch": parts[0],
        # "meta_package": parts[8],     # use for labels / grouping, NOT as a feature
        # "meta_program": parts[9],
    }


# main loop
# optional command-line arguments: only process these binaries (folder names)
only = sys.argv[1:]

rows = []
for ttl_file in sorted(TTL_DIR.glob(f"*/{TTL_NAME}")):
    binary_folder = ttl_file.parent.name          # the binary's name
    if only and binary_folder not in only:
        continue
    print("Processing", binary_folder)
    graph = Graph()
    graph.parse(ttl_file, format="turtle")

    row = {"file": binary_folder}

    # binary name is stored on the BinaryNode in the graph
    name_query = "SELECT ?n WHERE { ?b rdf:type :BinaryNode ; :hasName ?n } LIMIT 1"
    name_rows = list(graph.query(name_query, initNs=PREFIXES))
    if name_rows:
        binary_name = str(name_rows[0][0])
    else:
        # no BinaryNode with a name in this graph, so fall back to the folder name (which is the program name)
        print(f"WARNING: no BinaryNode name found in {ttl_file}, using folder name instead")
        binary_name = binary_folder
    row.update(filename_features(binary_name))

    # run every query and store its value under the query's name
    for column, query in queries.items():
        row[column] = run_query(graph, query)

    rows.append(row)

# then after running all the queries, write the results that are in rows to the csv file
# if no data exists, exit without writing anything to the csv file
if not rows:
    sys.exit(f"No {TTL_NAME} files found under {TTL_DIR}" + (f" for {only}" if only else ""))
# if the csv file has already been created, just append to it
if OUT_CSV.is_file():
    with OUT_CSV.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writerows(rows)
# else create the csv file, add the header, and add the data
else:
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
print(f"Wrote {len(rows)} rows to {OUT_CSV}")