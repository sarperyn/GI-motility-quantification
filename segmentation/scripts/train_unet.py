import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))

from torch.utils.data import DataLoader

from src.models.unet import get_configs, train_model
from src.utils.args_utils import train_arg_parser
from src.utils.model_utils import *
from src.utils.variable_utils import MADISON_STOMACH

if __name__ == '__main__':

    args = train_arg_parser()
    set_seed(args.seed)

    (model_class, dataset_class, 
    optimizer_class, optimizer_init_args, 
    scheduler_class, criterion, save_path) = get_configs(config_path=args.config)

    config_name = args.config.split('/')[-1].split('.')[0]
    save_path   = os.path.join(save_path, config_name)

    train_dataset = dataset_class(data_path=MADISON_STOMACH,
                            args=args, 
                            mode='train',
                            augment=True)

    val_dataset = dataset_class(data_path=MADISON_STOMACH,
                            args=args, 
                            mode='val',
                            augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False)

    print(f"LENGTH TRAIN:{len(train_dataset)}")
    print(f"LENGTH VAL:{len(val_dataset)}")

    model = model_class(in_channels=1, out_channels=1)
    model = model.to(args.device)

    optimizer = optimizer_class(model.parameters(), **optimizer_init_args)

    train_model(model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                args=args,
                save_path=save_path)
