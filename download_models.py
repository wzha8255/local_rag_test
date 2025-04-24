from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    local_dir="./models/all-MiniLM-L6-v2",
    local_dir_use_symlinks=False
)


snapshot_download(
    repo_id="microsoft/Phi-3-mini-4k-instruct",
    local_dir="./models/Phi-3",
    local_dir_use_symlinks=False
)