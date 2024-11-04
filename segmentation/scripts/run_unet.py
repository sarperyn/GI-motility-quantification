import os
import sys
import torch
from torch.utils.data import DataLoader

from src.models.unet import get_configs, train_model, test_model
from src.utils.args_utils import unet_arg_parser, save_arguments_to_file
from src.utils.model_utils import *
from src.utils.variable_utils import TRAIN_SET, VAL_SET

if __name__ == '__main__':

    # Parse arguments and save them to a file
    args = unet_arg_parser()
    save_arguments_to_file(args=args)
    set_seed(args.seed)

    # Load model, dataset, optimizer, scheduler, and criterion configurations
    (model_class, dataset_class, 
     optimizer_class, optimizer_init_args, 
     scheduler_class, criterion) = get_configs(config_path=args.config)

    # Set up the directory to save model checkpoints and logs
    save_path = os.path.join(args.save_dir, args.exp_id)

    # Initialize the model
    model = model_class(in_channels=1, out_channels=1)

    # Training mode
    if args.mode == 'train':
        
        # Use separate paths for training and validation datasets
        train_dataset = dataset_class(data_path=TRAIN_SET, args=args, mode='train')
        val_dataset = dataset_class(data_path=VAL_SET, args=args, mode='val')

        # Create DataLoaders for training and validation
        train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False)
        
        # Print dataset sizes for debugging
        print(f"LENGTH TRAIN: {len(train_dataset)}")
        print(f"LENGTH VAL: {len(val_dataset)}")

        # Move the model to the specified device (e.g., CPU or GPU)
        model = model.to(args.device)

        # Initialize the optimizer with the model's parameters
        optimizer = optimizer_class(model.parameters(), **optimizer_init_args)

        # Train the model
        train_model(model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    optimizer=optimizer,
                    criterion=criterion,
                    args=args,
                    save_path=save_path)
        
    # Testing mode
    elif args.mode == 'test':

        # Load the test dataset
        test_dataset = dataset_class(data_path=VAL_SET, args=args, mode='test')
        test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False) 

        # Load the saved model weights
        try:
            model.load_state_dict(torch.load(os.path.join(args.save_dir, args.exp_id, 'model', '*.pt'), map_location=args.device))
        except:
            model.load_state_dict(torch.load(args.model_path, map_location=args.device))
            
        # Test the model
        test_model(model=model, 
                   test_loader=test_loader, 
                   criterion=criterion, 
                   device=args.device, 
                   save_path=save_path, 
                   args=args)
