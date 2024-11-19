import os
import glob
import time
import subprocess
import concurrent.futures
import subprocess
import time
import glob
import concurrent.futures


def run_command(command):
    """Run a single shell command and handle errors."""
    print(f"Running: {command}")
    process = subprocess.run(command, shell=True)
    if process.returncode != 0:
        print(f"Error in: {command}")


def is_command_completed(session_name, check_phrase):
    """Check if the last command has completed in the tmux session."""
    # Capture the tmux pane output
    capture_command = f"tmux capture-pane -t {session_name} -p"
    result = subprocess.run(capture_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        print(f"Error capturing tmux pane for session {session_name}: {result.stderr}")
        return False

    # Check if the output contains the check_phrase
    pane_output = result.stdout
    return check_phrase in pane_output


def run_tmux_session_commands(session_name, commands, conda_env=None):
    """Run commands sequentially in a tmux session, waiting for each to complete."""
    # Create the tmux session if it doesn't exist
    new_session_command = f"tmux new-session -d -s {session_name}"
    run_command(new_session_command)
    time.sleep(3)  # Short delay to ensure session is created

    # Activate conda environment if provided
    if conda_env:
        activate_command = f"tmux send-keys -t {session_name} 'conda activate {conda_env}' Enter"
        run_command(activate_command)
        time.sleep(3)  # Delay to ensure environment activation

    # Send each command sequentially to the tmux session
    for cmd in commands:
        # Send the command to tmux
        tmux_command = f"tmux send-keys -t {session_name} '{cmd}' Enter"
        run_command(tmux_command)

        # Wait for the command to complete
        print(f"Waiting for command to complete in session {session_name}...")
        while not is_command_completed(session_name, "Completed"):
            time.sleep(5)  # Check every 5 seconds

        print(f"Command completed in session {session_name}: {cmd}")


def run_parallel_sessions(command_lists, conda_env=None):
    """Run multiple tmux sessions in parallel."""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(run_tmux_session_commands, f"singan_{i}", command_list, conda_env)
            for i, command_list in enumerate(command_lists)
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error in session: {e}")


def divide_list(input_list, n):
    """Divide a list into `n` approximately equal sublists."""
    sublist_size = len(input_list) // n
    remainder = len(input_list) % n

    sublists = []
    start = 0

    for i in range(n):
        end = start + sublist_size + (1 if i < remainder else 0)
        sublists.append(input_list[start:end])
        start = end

    return sublists


def create_commands(sublists, gpu_id, n_samples):
    """Create a list of Python commands for each sublist."""
    subcommands_list = []
    for sublist in sublists:
        subcommands = [
            f"python main_train.py --input_name {image.split('/')[-1]} --nc_z 4 --nc_im 4 --gpu_id {gpu_id} --n_samples {n_samples}"
            for image in sublist
        ]
        subcommands_list.append(subcommands)

    return subcommands_list


def get_commands():
    """Generate lists of commands to execute in parallel tmux sessions."""
    n_samples = 50
    images = glob.glob("/home/syurtseven/GI-motility-quantification/generation/singan-seg/Input/madison4d/*.png")
    n_parallel = 2
    gpu_id = '0'

    sublists = divide_list(images, n_parallel)
    subcommands = create_commands(sublists, gpu_id, n_samples)

    return subcommands


if __name__ == '__main__':
    command_lists = get_commands()
    conda_env = "gsoc"  # Conda environment name
    run_parallel_sessions(command_lists, conda_env)

