import os
import sys
import glob
sys.path.append(os.path.dirname(os.getcwd()))

from torch.utils.data import DataLoader

from src.models.unet import get_configs, train_model, test_model
from src.utils.args_utils import unet_arg_parser, save_arguments_to_file
from src.utils.model_utils import *
from src.utils.variable_utils import MADISON_STOMACH

if __name__ == '__main__':

    args = unet_arg_parser()
    save_arguments_to_file(args=args)
    set_seed(args.seed)

    (model_class, dataset_class, 
    optimizer_class, optimizer_init_args, 
    scheduler_class, criterion) = get_configs(config_path=args.config)

    save_path  = os.path.join(args.save_dir, args.exp_id)

    model = model_class(in_channels=1, out_channels=1, base_channel=args.base_channel)
    
    if args.mode == 'train':
         
        train_dataset = dataset_class(data_path=MADISON_STOMACH,
                            args=args, 
                            mode='train')

        val_dataset = dataset_class(data_path=MADISON_STOMACH,
                            args=args, 
                            mode='val')

        train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
        val_loader   = DataLoader(val_dataset, batch_size=args.bs, shuffle=False)
    
        print(f"LENGTH TRAIN:{len(train_dataset)}")
        print(f"LENGTH VAL:{len(val_dataset)}")

        model = model.to(args.device)

        optimizer = optimizer_class(model.parameters(), **optimizer_init_args)

        train_model(model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    optimizer=optimizer,
                    criterion=criterion,
                    args=args,
                    save_path=save_path)
        
    elif args.mode == 'test':

        test_dataset = dataset_class(data_path=MADISON_STOMACH,
                            args=args, 
                            mode='test')
    
        test_loader  = DataLoader(test_dataset, batch_size=args.bs, shuffle=False) 

        try:
            model.load_state_dict(torch.load(args.model_path, map_location=args.device))
        except:
            model.load_state_dict(torch.load(sorted(glob.glob(os.path.join(args.save_dir, args.exp_id, 'model','*.pt')))[-1], map_location=args.device))

            
        test_model(model=model.to(args.device), 
                   test_loader=test_loader, 
                   criterion=criterion, 
                   device=args.device, 
                   save_path=save_path, 
                   args=args)


