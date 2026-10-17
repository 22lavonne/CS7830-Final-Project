import tempfile
import sys
import shutil
from pathlib import Path
import pyghidra


# ghidrassist_path = Path("../opt/ghidra_12.1.4_PUBLIC_20260921/ghidra_12.1.4_PUBLIC/Extensions/Ghidra/ghidra_12.1_PUBLIC_20260530_GhidrAssist.zip")
# path to ghidrassist directory
# TODO: change this so it points to the submodule instead of a hard coded path based on just my system
ghidrassist_path = Path.home() / "GhidrAssist"

print("Checking path:", ghidrassist_path)
print("Does path exist?", ghidrassist_path.exists())
print("Contents of path:", [x.name for x in ghidrassist_path.iterdir() if x.is_dir()])

if str(ghidrassist_path) not in sys.path:
    sys.path.insert(0, str(ghidrassist_path))

from ghidrassist.graphrag import BinaryKnowledgeGraph
from ghidrassist import AnalysisDB

# starting ghidra headless using a hard coded path to the install directory
# TODO: change this to take in a user path for this instead of hard coding
pyghidra.start(install_dir="../opt/ghidra_12.1.4_PUBLIC_20260921/ghidra_12.1.4_PUBLIC")

# hard coding other paths needed
install_dir = Path("../opt/ghidra_12.1.4_PUBLIC_20260921/ghidra_12.1.4_PUBLIC")
script_dir = Path("scripts_for_ghidra/combined_extraction.py")

# TODO: change these so they go in a loop for all the files in the directory
binary_path = Path("test-files/ARM__32__lsb__unix-system-v__arm-linux-gnueabihf-gcc-5.4.0__O0__no-obf__unstripped__coreutils-8.30__chcon")
# project_dir = Path("/home/emiller/CS7830-Final-Project/test-files/ARM__32__lsb__unix-system-v__arm-linux-gnueabihf-gcc-5.4.0__O0__no-obf__unstripped__coreutils-8.30__chcon_ghidra/ARM__32__lsb__unix-system-v__arm-linux-gnueabihf-gcc-5.4.0__O0__no-obf__unstripped__coreutils-8.30__chcon_ghidra.rep")

# this removes the temporary directory that pyghidra creates whenever it opens a project
companion_project = Path(f"{binary_path}_ghidra")
if companion_project.exists():
    shutil.rmtree(companion_project)
    
# get a new project name and put it in a different directory containing all the projects (based on executable name)
project_dir = binary_path.parent / "projects"
project_name = f"{binary_path.name}_ghidra_proj"
    
# then create this project if it doesn't already exist
project = pyghidra.open_project(project_dir, project_name, create=True)

try:
    # open the current proram (open_program is depricated, but it still works)
    with pyghidra.open_program(str(binary_path)) as pyghidra_program:
        program = pyghidra_program.getCurrentProgram()
        print(f"Successfully loaded: {program.getName()}")
        print(f"Architecture: {program.getLanguageID()}\n")

        # # testing by listing all the functions
        # function_manager = program.getFunctionManager()
        # print("Listing functions found in binary:")
        # for func in function_manager.getFunctions(True):
        #     print(f" - Found function: {func.getName()} at {func.getEntryPoint()}")

        
        
        try:
            print("Triggering programmatic GhidrAssist graph indexing...")
            db = AnalysisDB()
            program_hash = program.getExecutableSHA256()
            
            # Check if graph exists or trigger generation/sync
            graph = db.getKnowledgeGraph(program_hash)
            if not graph or len(graph.getNodesByType(None)) == 0:
                print("Graph data missing for hash. Initializing build...")
                # If your version uses a specific builder utility, call it here:
                # e.g., BinaryKnowledgeGraph.buildForProgram(program, monitor)
            
        except Exception as e:
            print(f"Note on programmatic indexing hook: {e}")
            
        # running the script that extracts the data needed from the executable into json files
        pyghidra.ghidra_script(path=script_dir, program=program, project=project)
            
finally:
    project.close()
    