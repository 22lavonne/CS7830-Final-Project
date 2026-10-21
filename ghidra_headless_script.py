import sys
import os
import shutil
import zipfile
from pathlib import Path
import pyghidra
import subprocess


# ---------- hard coded paths ----------
# install directory for ghidra
INSTALL_DIR = Path("../opt/ghidra_12.1.4_PUBLIC_20260921/ghidra_12.1.4_PUBLIC").resolve()

# path for where the ghidrassist zip extension is (within the )
GHIDRASSIST_ZIP = Path(
    INSTALL_DIR / "Extensions/Ghidra/ghidra_12.1_PUBLIC_20260530_GhidrAssist.zip"
).resolve()

# path for the extraction script
SCRIPT_PATH = Path("scripts_for_ghidra/combined_extraction.py")


# semantic analysis in ghidrassist does the llm summaries, which takes forever
# so we turn that off, as we do not need it for the features we are extracting.
RUN_SEMANTIC_ANALYSIS = False
SEMANTIC_MAX_NODES = 100   # max functions to summarize per binary
# TODO: change this so it loops through all the files in the test-files directory instead of just one
BINARY_PATH = Path(
    "test-files/ARM__32__lsb__unix-system-v__arm-linux-gnueabihf-gcc-5.4.0__O0__"
    "no-obf__unstripped__coreutils-8.30__chcon"
)

# function to get the directory where extensions go based on the install directory
def ghidra_user_extension_dir(install_dir: Path) -> Path:
    props = {}
    for line in (install_dir / "Ghidra" / "application.properties").read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            props[k.strip()] = v.strip()
    version = props["application.version"]          # e.g. 12.1.4
    release = props["application.release.name"]     # e.g. PUBLIC
    return Path.home() / ".config" / "ghidra" / f"ghidra_{version}_{release}" / "Extensions"

# function to install ghidrassist into the user's extensions folder (if it isn't there already)
def install_extension(zip_path: Path, install_dir: Path) -> Path:
    ext_root = ghidra_user_extension_dir(install_dir)
    target = ext_root / "GhidrAssist"
    if not target.exists():
        print(f"Installing {zip_path.name} -> {ext_root}")
        ext_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            # the top-level folder here is GhidrAssist
            zf.extractall(ext_root)
    else:
        print(f"GhidrAssist already installed at {target}")
    if not target.exists():
        sys.exit(f"Expected {target} after extraction; check the zip's folder layout.")
    return target

# this actually installs the extension (if not already installed)
ext_dir = install_extension(GHIDRASSIST_ZIP, INSTALL_DIR)

# we can start pyghidra in the headless version, based on the installation directory of ghidra
pyghidra.start(install_dir=str(INSTALL_DIR))

# these imports must be here because they are only available after pyghidra has started in headless mode.
# (will otherwise result in an import error if they are at the top of the file)
# ghidra import:
from ghidra.util.task import TaskMonitor
# ghidrassist imports:
from ghidrassist import AnalysisDB
from ghidrassist.graphrag import GraphRAGService
from ghidrassist.graphrag.community import CommunityDetector

# if there is an instance of this file, then it needs to be removed
# since it interferes with the number of nodes from the executable
# (it generates every time ghidrassist reindexes a new binary)
prev_database = Path("ghidrassist_analysis.db")
if prev_database.exists():
    os.remove(prev_database)

# this removes the temporary directory that pyghidra creates whenever it opens a project
companion_project = Path(f"{BINARY_PATH}_ghidra")
if companion_project.exists():
    shutil.rmtree(companion_project)

# getting the project directory and opening the project
project_dir = BINARY_PATH.parent / "projects"
project_name = f"{BINARY_PATH.name}_ghidra_proj"
project = pyghidra.open_project(project_dir, project_name, create=True)

# get the name of the program so the materialization script later knows what directory to look for for the files
program_name = None

try:
    with pyghidra.open_program(str(BINARY_PATH)) as flat_api:
        program = flat_api.getCurrentProgram()
        print(f"Successfully loaded: {program.getName()}")
        print(f"Architecture: {program.getLanguageID()}\n")
        
        program_name = str(program.getName())

        monitor = TaskMonitor.DUMMY
        db = AnalysisDB()
        service = GraphRAGService.getInstance(db)
        service.setCurrentProgram(program)

        # this "reindexes" the binary, which is what ghidrassist uses to populate the semantic graph
        print("[GhidrAssist] ReIndex Binary...")
        try:
            result = service.indexStructureSync(program, monitor, False)
            print(f"[GhidrAssist] Reindex result: {result}")
        except Exception as e:
            print(f"[GhidrAssist] Reindex FAILED: {type(e).__name__}: {e}")

        # printing here to check for stats, but uneeded when all the features will be extracted into separate files
        # print("[GhidrAssist] Graph stats:", service.getGraphStats(program))

        # the next few try except blocks run the different types of analysis from ghidrassist
        # which happens after the binary is reindexed.
        
        # security analysis:
        print("[GhidrAssist] Security Analysis...")
        try:
            n = service.updateAllSecurityFlags(program, monitor)
            print(f"[GhidrAssist] Security flags updated for {n} functions")
        except Exception as e:
            print(f"[GhidrAssist] Security analysis FAILED: {type(e).__name__}: {e}")

        # community detection (if functions have been identified to be part of a community)
        try:
            binary_id = program.getExecutableSHA256()
            # print(f"[GhidrAssist] Binary id (SHA256): {binary_id}")
            graph = db.getKnowledgeGraph(binary_id)
            if graph is not None:
                print("[GhidrAssist] Community Detection...")
                found = CommunityDetector(graph, monitor).detectCommunities()
                print(f"[GhidrAssist] Communities found: {found}")
                print(f"[GhidrAssist] Nodes: {graph.getNodeCount()}, "
                      f"edges: {graph.getEdgeCount()}")
            else:
                print("[GhidrAssist] No graph returned for that binary id")
        except Exception as e:
            print(f"[GhidrAssist] Community detection FAILED: {type(e).__name__}: {e}")

        # run your existing extraction script
        pyghidra.ghidra_script(path=SCRIPT_PATH, program=program, project=project)

finally:
    project.close()
    
# after running the pyghidra script, now we want to materialize the knowledge graph 
# based on the json files that were created from this
# run the other python file as a subprocess
result = subprocess.run([sys.executable, "scripts_for_ghidra/combined_materialization.py", program_name])
print("Materialization script finished with return code of", result.returncode)

# then we want to run the script that writes to the csv file with the extracted features
# if result.returncode == 0:
#     build = subprocess.run([sys.executable, "scripts_for_ghidra/build_dataset_single.py"])
#     print("Dataset builder finished with return code of", build.returncode)
# else:
#     print("Materialization failed, skipping dataset build")

if result.returncode == 0:
    build = subprocess.run([sys.executable, "scripts_for_ghidra/dataset_builder.py", program_name])
    print("Dataset builder finished with return code of", build.returncode)