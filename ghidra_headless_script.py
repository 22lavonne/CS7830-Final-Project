import tempfile
import shutil
from pathlib import Path
import pyghidra

# 1. Start Ghidra headless JVM
pyghidra.start(install_dir="../opt/ghidra_12.1.4_PUBLIC_20260921/ghidra_12.1.4_PUBLIC")

binary_path = Path("test-files/ARM__32__lsb__unix-system-v__arm-linux-gnueabihf-gcc-5.4.0__O0__no-obf__unstripped__coreutils-8.30__chcon")

# this removes the temporary directory that pyghidra creates whenever it opens a project
companion_project = Path(f"{binary_path}_ghidra")
if companion_project.exists():
    shutil.rmtree(companion_project)

# # 2. Use a temporary directory for the project so you don't get stuck with lock files
# with tempfile.TemporaryDirectory() as tmpdir:
#     # Open or create a temporary project
#     project = pyghidra.open_project(tmpdir, "my_temp_project", create=True)
    
    
    # 3. Open the program headlessly and list all functions
with pyghidra.open_program(str(binary_path)) as flat_api:
    program = flat_api.getCurrentProgram()
    print(f"Successfully loaded: {program.getName()}")
    print(f"Architecture: {program.getLanguageID()}\n")

    function_manager = program.getFunctionManager()
    print("Listing functions found in binary:")
    for func in function_manager.getFunctions(True):
        print(f" - Found function: {func.getName()} at {func.getEntryPoint()}")
    # Programmatically import the binary into the project path
    # (Or use project management if you prefer saving a permanent project folder)
    
    # Alternatively, if you want a permanent project folder next to your script:
    # project = pyghidra.open_project(Path.cwd(), "ghidra_project", create=True)

    # For now, let's use the explicit project context loader:
    # (If the binary is already imported into the project, you use program_context)