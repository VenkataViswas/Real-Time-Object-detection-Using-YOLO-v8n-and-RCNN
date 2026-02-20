from huggingface_hub import snapshot_download

# Download dataset
snapshot_download(
    repo_id="crangana/pedestrian_data",
    repo_type="dataset",
    local_dir="./data"
)

print("Dataset downloaded successfully!")