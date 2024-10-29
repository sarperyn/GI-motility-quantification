import os
import glob
import subprocess
import time


def run_generation():

    n_samples = 50
    images = glob.glob("/home/syurtseven/gsoc-2024/external/singan-seg/Input/madison4d/*.png")
    n_parallel = 4  
    gpu_id = 'cuda:0'
    conda_env = "gsoc"  

    session_names = create_tmux_terminals(n_parallel=n_parallel, conda_env=conda_env)

    for idx, image_path in enumerate(images):

        img_name = os.path.basename(image_path)
        python_commands = get_batch_commands(n_parallel, img_name, n_samples, gpu_id)
        run_tmux_commands(python_commands, session_names)

        time.sleep(2)


def create_tmux_terminals(n_parallel, conda_env):

    session_names = []
    for i in range(n_parallel):
        session_name = f"singan_{i}"
        session_names.append(session_name)
        command1 = f'tm+new-session -d -s {session_name}'
        subprocess.run(command1, shell=True)

        command2 = f'tmux send-keys -t {session_name} conda activate {conda_env}+Enter'
        subprocess.run(command2, shell=True)

    return session_names

def get_batch_commands(n_parallel, img_name, n_samples, gpu_id):

    python_commands = []
    for i in range(n_parallel):
        python_commands.append(f"python main_train.py --input_name {img_name} --nc_z 4 --nc_im 4 --gpu_id {gpu_id} --n_samples {n_samples}")

    return python_commands

def run_tmux_commands(python_commands, session_names):

    for i in range(len(python_commands)):

        command1 = f'tmux attach -t {session_names[i]}+Enter'
        subprocess.run(command1.split('+'), shell=True)
        subprocess.run(python_commands[i])




if __name__ == '__main__':
    run_generation()