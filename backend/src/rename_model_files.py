import os
from pathlib import Path

def rename_model_files():
    model_dir = Path("models/mixtral_local")
    
    # Get all files with "(1)" in their name
    files_to_rename = [f for f in model_dir.glob("* (1)*")]
    
    for file_path in files_to_rename:
        # Create new name by removing " (1)"
        new_name = str(file_path).replace(" (1)", "")
        try:
            os.rename(str(file_path), new_name)
            print(f"Renamed: {file_path.name} -> {Path(new_name).name}")
        except Exception as e:
            print(f"Error renaming {file_path.name}: {str(e)}")

if __name__ == "__main__":
    rename_model_files() 