import os
import shutil

# --- CONFIGURATION ---
# 1. The main folder that CONTAINS all your smaller folders with pictures.
SOURCE_FOLDER = r"F:\Learning\Data Analysic\Convolutional-Neural-Networks\PetImages\CUB_200_2011\images"

# 2. The new folder WHERE you want to put all the pictures.
DESTINATION_FOLDER = r"F:\All My Pictures"

# 3. List of image file extensions you want to find.
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
# --------------------

# Create the destination folder if it doesn't exist
if not os.path.exists(DESTINATION_FOLDER):
    os.makedirs(DESTINATION_FOLDER)
    print(f"Created destination folder: {DESTINATION_FOLDER}")

# os.walk() recursively goes through every folder and file in the source folder
for root_folder, subfolders, filenames in os.walk(SOURCE_FOLDER):
    for filename in filenames:
        # Check if the file has one of the desired image extensions
        if any(filename.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
            
            # Create the full path to the source file
            source_path = os.path.join(root_folder, filename)
            
            # Create the path for the destination file
            destination_path = os.path.join(DESTINATION_FOLDER, filename)
            
            # Simple check to avoid errors if a file with the same name already exists
            if not os.path.exists(destination_path):
                # Copy the file. shutil.copy2 also copies metadata like creation date.
                shutil.copy2(source_path, destination_path)
                print(f"Copied: {filename}")
            else:
                print(f"Skipped (already exists): {filename}")

print("\n--- Process Complete! ---")