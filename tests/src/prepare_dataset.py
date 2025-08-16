import os
import random
import shutil
from huggingface_hub import hf_hub_download
import pandas as pd

def main():
    sate_iv_dir = "data/sate_iv"
    output_dir = "data/processed"
    
    print("Creating output directories...")
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)
    
    all_files = []

    print("Processing CVEFixes dataset...")
    try:
        cve_file_path = hf_hub_download(repo_id="MickyMike/cvefixes_bigvul", filename="train.csv", repo_type="dataset", local_dir="data")
        cve_df = pd.read_csv(cve_file_path)
        for index, row in cve_df.iterrows():
            code = row['target']
            # Check if code is a string
            if isinstance(code, str):
                file_path = os.path.join(output_dir, f"cve_{row['cve_id']}_{index}.py")
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(code)
                all_files.append(file_path)
    except Exception as e:
        print(f"Error processing CVEFixes: {e}")

    print("Processing SATE IV dataset...")
    for root, _, files in os.walk(sate_iv_dir):
        for file in files:
            if file.endswith(".c"):
                all_files.append(os.path.join(root, file))
    
    print("Processing CodeSearchNet dataset...")
    try:
        csn_file_path = hf_hub_download(repo_id="AhmedSSoliman/CodeSearchNet", filename="train.csv", repo_type="dataset", local_dir="data")
        csn_df = pd.read_csv(csn_file_path)
        for index, row in csn_df.iterrows():
            code = row['code']
            if isinstance(code, str):
                file_path = os.path.join(output_dir, f"csn_{index}.py")
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(code)
                all_files.append(file_path)
    except Exception as e:
        print(f"Error processing CodeSearchNet: {e}")

    print("Shuffling and splitting dataset...")
    random.shuffle(all_files)
    
    train_split = 0.7
    val_split = 0.15
    
    train_files = all_files[:int(len(all_files) * train_split)]
    val_files = all_files[int(len(all_files) * train_split):int(len(all_files) * (train_split + val_split))]
    test_files = all_files[int(len(all_files) * (train_split + val_split)):]
    
    print("Copying files to output directories...")
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")

    for file in train_files:
        try:
            shutil.copy(file, train_dir)
        except Exception as e:
            print(f"Error copying {file} to train: {e}")
        
    for file in val_files:
        try:
            shutil.copy(file, val_dir)
        except Exception as e:
            print(f"Error copying {file} to val: {e}")
        
    for file in test_files:
        try:
            shutil.copy(file, test_dir)
        except Exception as e:
            print(f"Error copying {file} to test: {e}")

    print("Cleaning up temporary files...")
    for file in all_files:
        if os.path.exists(file) and output_dir in file:
            os.remove(file)
            
    print("Dataset preparation complete.")

if __name__ == "__main__":
    main()
