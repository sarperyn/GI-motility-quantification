import subprocess
########################################################################
########### MULTI EXPERIMENT SCRIPT ####################################
########################################################################
# This script runs multiple experiments in parallel using `tmux`.
# Each experiment is executed in a separate `tmux` session, enabling
# the parallel execution of Python scripts with different configurations.
########################################################################

# List of Python commands for running experiments with different configurations
exp_h = [
    # Experiment 1: Base channel = 2, running on CUDA device 0
    "python run_unet.py --mode train --config /home/syurtseven/GI-motility-quantification/segmentation/config/experiments/MadisonUNet.yaml --exp_id complexity/0 --device cuda:0 --bs 5 --epoch 20 --base_channel 2",
    # Experiment 2: Base channel = 8, running on CUDA device 0
    "python run_unet.py --mode train --config /home/syurtseven/GI-motility-quantification/segmentation/config/experiments/MadisonUNet.yaml --exp_id complexity/1 --device cuda:0 --bs 5 --epoch 20 --base_channel 8",
    # Experiment 3: Base channel = 64, running on CUDA device 0
    "python run_unet.py --mode train --config /home/syurtseven/GI-motility-quantification/segmentation/config/experiments/MadisonUNet.yaml --exp_id complexity/3 --device cuda:0 --bs 5 --epoch 20 --base_channel 64",
    # Experiment 4: Base channel = 200, running on CUDA device 1
    "python run_unet.py --mode train --config /home/syurtseven/GI-motility-quantification/segmentation/config/experiments/MadisonUNet.yaml --exp_id complexity/5 --device cuda:1 --bs 5 --epoch 20 --base_channel 200",
]

def main():
    """
    Main function to create `tmux` sessions and run the experiments.
    Each experiment runs in a separate `tmux` session to enable parallel execution.
    """
    bashcode = ''  # Placeholder for additional bash commands (if needed)
    python_commands = exp_h  # List of experiment commands

    # Dictionary to map unique session names to the corresponding experiment commands
    commands = {
        f"exp-{i}": bashcode + el for i, el in enumerate(python_commands)
    }
    
    # Command to initialize conda in `tmux`
    eval = 'eval "$(conda shell.bash hook)"'

    # Loop through each experiment and create its corresponding `tmux` session
    for k, v in commands.items():
        # Create a new `tmux` session with a unique name
        code1 = f"tmux+new-session+-d+-s+{k}"
        # Send the command to initialize conda in the `tmux` session
        code2 = f"tmux+send-keys+-t+{k}+{eval}+Enter"
        # Activate the specific conda environment in the session
        code3 = f"tmux+send-keys+-t+{k}+conda activate gsoc+Enter"
        # Run the experiment command in the session
        code4 = f"tmux+send-keys+-t+{k}+{v}+Enter"

        # Execute the commands sequentially
        for i in [code1, code2, code3, code4]:
            res = subprocess.run(i.split('+'))  # Run each command using `subprocess`
            print(res)  # Print the result of the command execution

# Entry point of the script
if __name__ == '__main__':
    main()
