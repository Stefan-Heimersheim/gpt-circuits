from huggingface_hub import snapshot_download

# Replace with your desired target folder
local_dir = "fineweb_edu_10b"

# Download the entire repository snapshot (including LFS files)
snapshot_download(
    repo_id="davidquarel/fineweb_edu_10b",
    repo_type="model",  # or "dataset" if it's hosted as a dataset
    local_dir=local_dir,
    local_dir_use_symlinks=False,  # Store real files, not symlinks
    resume_download=True  # Good for big files or interrupted downloads
)

print(f"✅ All files downloaded to: {local_dir}")
