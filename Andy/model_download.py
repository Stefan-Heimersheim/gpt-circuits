from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
from huggingface_hub import upload_file



sparsities = ['1.0e-01', '1.0e-02','1.0e-03','1.0e-04','1.0e-05', '1.0e-06', 
                '1.5e-01', '1.5e-02', '1.5e-03', '1.5e-04', '1.5e-05', '1.5e-06', 
                '2.2e-01', '2.2e-02', '2.2e-03', '2.2e-04', '2.2e-05', '2.2e-06',
                '3.3e-01', '3.3e-02', '3.3e-03', '3.3e-04', '3.3e-05', '3.3e-06',
                '4.7e-01', '4.7e-02', '4.7e-03', '4.7e-04', '4.7e-05', '4.7e-06',
                '6.8e-01', '6.8e-02', '6.8e-03', '6.8e-04', '6.8e-05', '6.8e-06', '0.0ep00']


#sparsities = ['1.0e-04', '1.0e-05', '1.0e-06', '1.0e-07', '1.0e-08', '1.0ep00', '1.0ep01', 
#'3.3e-02', '3.3e-04', '3.3e-05', '3.3e-06', '3.3e-07', '3.3e-08', '3.3ep00', '3.3e-01', '0.0ep00']

model_names = [f'davidquarel/jblock.shk_64x4-sparse-{sparsity}' for sparsity in sparsities]
local_dirs = [f'checkpoints/jblock.shk_64x4-sparse-{sparsity}' for sparsity in sparsities]
#configs = ['jsae.topkx8.shakespeare_64x4' for sparsity in sparsities]
save_names = [f'jblock.shk_64x4-sparse-{sparsity}' for sparsity in sparsities]
batch_size = 24
num_batches = 24
steps=5
download = True

with open('commands.txt', 'w') as f:
    
    for idx, model_name in enumerate(model_names):
        local_dir = local_dirs[idx]
        sparsity = sparsities[idx]

        name = save_names[idx] 

        if download:
            snapshot_download(repo_id=model_name, local_dir=local_dir, local_dir_use_symlinks=False)
            print(f"Model downloaded successfully into {local_dir}")
            bash_command = f'python Andy/compute_attributions.py --load_from={local_dir} --save_to=Andy/data --data_dir=data/shakespeare --save_name={name} --num_batches={num_batches} --batch_size={batch_size} --steps={steps} --attribution_method=ig'
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
