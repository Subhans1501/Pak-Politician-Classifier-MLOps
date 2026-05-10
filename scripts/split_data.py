import splitfolders
import os

def create_dataset_splits():
    input_dir = "dataset/raw"
    output_dir = "dataset/split"
    print(f"Reading raw images from: {input_dir}")
    print("Splitting dataset into 75% Train, 15% Val, and 10% Test...")
    splitfolders.ratio(
        input_dir, 
        output=output_dir, 
        seed=42, 
        ratio=(0.75, 0.15, 0.10), 
        group_prefix=None, 
        move=False
    )
    
    print(f"Data successfully split and saved to: {output_dir}")

if __name__ == "__main__":
    if not os.path.exists("dataset/raw"):
        print("Error: 'dataset/raw' folder not found. Please add your images first.")
    else:
        create_dataset_splits()