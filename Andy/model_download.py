from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
from huggingface_hub import upload_file



sparsities = ['1.0e-01']
#sparsities = ['1.0e-04', '1.0e-05', '1.0e-06', '1.0e-07', '1.0e-08', '1.0ep00', '1.0ep01', 
#'3.3e-02', '3.3e-04', '3.3e-05', '3.3e-06', '3.3e-07', '3.3e-08', '3.3ep00', '3.3e-01', '0.0ep00']

model_names = [f'davidquarel/jblock.shk_64x4-sparse-{sparsity}' for sparsity in sparsities]
local_dirs = [f'checkpoints/jblock.shk_64x4-sparse-{sparsity}' for sparsity in sparsities]
configs = ['jsae.topkx8.shakespeare_64x4' for sparsity in sparsities]
save_names = [f'jblock.shk_64x4-sparse-{sparsity}' for sparsity in sparsities]

download = True

with open('commands.txt', 'w') as f:
    
    for idx, model_name in enumerate(model_names):
        local_dir = local_dirs[idx]
        sparsity = sparsities[idx]
        config = configs[idx]
        name = save_names[idx] 

        if download:
            snapshot_download(repo_id=model_name, local_dir=local_dir, local_dir_use_symlinks=False)
            print(f"Model downloaded successfully into {local_dir}")
            bash_command = f'python Andy/compute_attributions.py --config={config} --load_from={local_dir} --save_to=Andy/data --data_dir=data/shakespeare --save_name={name} --num_batches=2 --batch_size=2 --attribution_method=ig'
            f.write(bash_command + '\n')
        else:
            repo_id = "algo2217/SPAR-attributions"
            path = f"Andy/data/{name}"

            upload_file(
                path_or_fileobj=path,
                path_in_repo=f"{name}",
                repo_id=repo_id,
                repo_type="dataset"
            )
            print(f"uploaded file at {path}")
