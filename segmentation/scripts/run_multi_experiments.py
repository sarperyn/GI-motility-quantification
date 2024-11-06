import subprocess
########################################################################
########################################################################
######## MULTI EXPERIMENT SCRIPT #######################################
########################################################################
########################################################################

exp_h = [
    "python run_unet.py --mode train --config /home/syurtseven/GI-motility-quantification/segmentation/config/experiments/MadisonUNet.yaml --exp_id complexity/0 --device cuda:0 --bs 5 --epoch 20 --base_channel 2",
    "python run_unet.py --mode train --config /home/syurtseven/GI-motility-quantification/segmentation/config/experiments/MadisonUNet.yaml --exp_id complexity/1 --device cuda:0 --bs 5 --epoch 20 --base_channel 8",
    "python run_unet.py --mode train --config /home/syurtseven/GI-motility-quantification/segmentation/config/experiments/MadisonUNet.yaml --exp_id complexity/3 --device cuda:0 --bs 5 --epoch 20 --base_channel 64",
    "python run_unet.py --mode train --config /home/syurtseven/GI-motility-quantification/segmentation/config/experiments/MadisonUNet.yaml --exp_id complexity/5 --device cuda:1 --bs 5 --epoch 20 --base_channel 200",

]

def main():
    
    bashcode = ''
    python_commands = exp_h

    commands = {
        f"exp-{i}":bashcode + el for i,el in enumerate(python_commands)
    }
    
    eval = 'eval "$(conda shell.bash hook)"'
    for k,v in commands.items():
        
        code1 = f"tmux+new-session+-d+-s+{k}"
        code2 = f"tmux+send-keys+-t+{k}+{eval}+Enter"
        code3 = f"tmux+send-keys+-t+{k}+conda activate gsoc+Enter"
        code4 = f"tmux+send-keys+-t+{k}+{v}+Enter"

        for i in [code1, code2, code3, code4]:
            res = subprocess.run(i.split('+'))
            print(res)


if __name__ == '__main__':
    main()