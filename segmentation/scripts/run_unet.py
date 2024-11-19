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
    # Parse command-line arguments for the U-Net model
    args = unet_arg_parser()

    # Save the parsed arguments to a file for reproducibility
    save_arguments_to_file(args=args)

    # Set the random seed for reproducibility
    set_seed(args.seed)

    # Retrieve configurations for the model, dataset, optimizer, scheduler, and loss function
    (model_class, dataset_class, 
     optimizer_class, optimizer_init_args, 
     scheduler_class, criterion) = get_configs(config_path=args.config)

    # Define the directory path to save experiment outputs
    save_path = os.path.join(args.save_dir, args.exp_id)

    # Initialize the U-Net model with the specified number of base channels
    model = model_class(in_channels=1, out_channels=1, base_channel=args.base_channel)
    
    # Training mode
    if args.mode == 'train':
        # Initialize the training and validation datasets
        train_dataset = dataset_class(data_path=MADISON_STOMACH, args=args, mode='train')
        val_dataset = dataset_class(data_path=MADISON_STOMACH, args=args, mode='val')

        # Create DataLoaders for training and validation
        train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False)
    
        # Print the sizes of the training and validation datasets
        print(f"LENGTH TRAIN: {len(train_dataset)}")
        print(f"LENGTH VAL: {len(val_dataset)}")

        # Move the model to the specified device (CPU or GPU)
        model = model.to(args.device)

        # Initialize the optimizer with model parameters and arguments
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
        # Initialize the test dataset
        test_dataset = dataset_class(data_path=MADISON_STOMACH, args=args, mode='test')

        # Create a DataLoader for testing
        test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False)

        # Load the trained model's weights
        try:
            # Load the model from a specified file path
            model.load_state_dict(torch.load(args.model_path, map_location=args.device))
        except:
            # Load the most recent model checkpoint from the save directory
            model.load_state_dict(torch.load(sorted(glob.glob(os.path.join(args.save_dir, args.exp_id, 'model', '*.pt')))[-1], map_location=args.device))

        # Test the model on the test dataset
        test_model(model=model.to(args.device),
                   test_loader=test_loader,
                   criterion=criterion,
                   device=args.device,
                   save_path=save_path,
                   args=args)
