from huggingface_hub import list_repo_files

files = list_repo_files(repo_id="MickyMike/cvefixes_bigvul", repo_type="dataset")
print(files)