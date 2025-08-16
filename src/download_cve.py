from huggingface_hub import hf_hub_download

hf_hub_download(repo_id="MickyMike/cvefixes_bigvul", filename="cvefixes_bigvul.csv", repo_type="dataset", local_dir="data")
