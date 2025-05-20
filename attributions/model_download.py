from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
from huggingface_hub import upload_file



sparsities = ['1.0e-01', '1.0e-02','1.0e-03','1.0e-04','1.0e-05', '1.0e-06', 
                '1.5e-01', '1.5e-02', '1.5e-03', '1.5e-04', '1.5e-05', '1.5e-06', 
                '2.2e-01', '2.2e-02', '2.2e-03', '2.2e-04', '2.2e-05', '2.2e-06',
                '3.3e-01', '3.3e-02', '3.3e-03', '3.3e-04', '3.3e-05', '3.3e-06',
                '4.7e-01', '4.7e-02', '4.7e-03', '4.7e-04', '4.7e-05', '4.7e-06',
                '6.8e-01', '6.8e-02', '6.8e-03', '6.8e-04', '6.8e-05', '6.8e-06', '0.0ep00']


#Edit these!
model_names = [f'HF_REPO/model_name-{sparsity}' for sparsity in sparsities] #A list of existing models on huggingface
local_dirs = [f'checkpoints/model_name-{sparsity}' for sparsity in sparsities] #This could be anywhere you want the model saved to
save_names = ['model_name-{sparsity}.safetensors' for sparsity in sparsities] #This could be any name, but it should be unique for each run you want

download = True #If you set this to false, the same file will upload the attributions instead of downloading models. Don't forget to log in to huggingface!

#These are reasonable parameters, and what I've been using for all the data. You shouldn't need to modify
batch_size = 24
num_batches = 24
steps=5
save_repo_id = "Huggingface repo to save attributions to"


with open('commands.txt', 'w') as f:
    
    for idx, model_name in enumerate(model_names):
        local_dir = local_dirs[idx]
        sparsity = sparsities[idx]

        name = save_names[idx] 

        if download:
            snapshot_download(repo_id=model_name, local_dir=local_dir, local_dir_use_symlinks=False)
            print(f"Model downloaded successfully into {local_dir}")
            bash_command = f'python -m attributions.compute_attributions --load_from={local_dir} --save_to=attributions/data --data_dir=data/shakespeare --save_name={name} --num_batches={num_batches} --batch_size={batch_size} --steps={steps} --attribution_method=ig'
            f.write(bash_command + '\n')
        else:
            repo_id = save_repo_id
            path = f"attributions/data/{name}"

            upload_file(
                path_or_fileobj=path,
                path_in_repo=f"{name}",
                repo_id=repo_id,
                repo_type="dataset"
            )
            print(f"uploaded file at {path}")
