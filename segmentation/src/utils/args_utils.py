import argparse
import os
import json
import argparse

def unet_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda:0', 
                        help="Device to use for computation. Set to 'cuda:0' for GPU or 'cpu' for CPU.")

    parser.add_argument('--exp_id', type=str, default='exp/0', 
                        help="Identifier for the experiment, used to create subdirectories for results.")

    parser.add_argument('--wandb', action='store_true', 
                        help="If set, enables logging with Weights & Biases (WandB) for experiment tracking.") # this is not working rn

    parser.add_argument('--seed', type=int, default=42, 
                        help="Random seed for reproducibility of results.")

    parser.add_argument('--lr', type=float, default=3e-3, 
                        help="Learning rate for the optimizer.")

    parser.add_argument('--bs', type=int, default=10, 
                        help="Batch size for training, determining the number of samples per gradient update.")

    parser.add_argument('--epoch', type=int, default=20, 
                        help="Number of epochs for training, the number of full dataset iterations.")

    parser.add_argument('--config', type=str, required=True, 
                        help="Path to the configuration file with additional settings for the experiment.")

    parser.add_argument('--save_dir', type=str, default='/home/syurtseven/GI-motility-quantification/segmentation/results/UNet', 
                        help="Directory where the results and model files will be saved. Change for your experiment directory.")

    parser.add_argument('--base_channel', type=int, default=64, 
                        help="The base number of channels/filters in the model layers. "
                         "Increasing this value will increase the model's complexity and parameter count, "
                         "while decreasing it will result in a smaller, lighter model. ")

    parser.add_argument('--model_path', type=str, default='', 
                        help="Path to a pre-trained model file to load for training or evaluation. Leave blank if not used.")
    
    parser.add_argument('--mode', type=str, required=True, 
                        help="Operational mode of the script. Set to 'train' for training or 'test' for evaluation.")

    args = parser.parse_args()
    return args


def svm_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_dir', type=str, default='/home/syurtseven/GI-motility-quantification/segmentation/results/SVM', 
                        help="Directory where the results and model files will be saved.")

    parser.add_argument('--exp_id', type=str, default='deneme/0', 
                        help="Identifier for the experiment, used to create subdirectories for results.")

    parser.add_argument('--kernel', type=str, default='rbf', 
                        help="Specifies the kernel type to be used in the SVM algorithm.")

    parser.add_argument('--gamma', type=float, default=0.001, 
                        help="Gamma parameter for the SVM kernel. Controls the influence of individual points.")

    parser.add_argument('--C', type=float, default=1, 
                        help="Regularization parameter for SVM. Higher values mean stricter fitting to the data.")

    parser.add_argument('--seed', type=int, default=42, 
                        help="Random seed for reproducibility of results.")

    parser.add_argument('--n_samples', type=int, default=30, 
                        help="Number of samples to be loaded from the dataset.")

    parser.add_argument('--mode', type=str, required=True, 
                        help="Operational mode of the script. Set to 'train' for training or 'test' for evaluation.")

    parser.add_argument('--target_resize', type=int, nargs=2, default=(128, 128), 
                        help="Target dimensions for resizing images, specified as width and height (e.g., 256 256).")

    parser.add_argument('--scale_method', type=str, default='standard', 
                        help="Scaling method for preprocessing the data (e.g., 'standard' for StandardScaler or 'minmax' MinMaxScaler).")

    parser.add_argument('--dim_reduction_method', type=str, default=None, 
                        help="Optional dimensionality reduction technique (e.g., 'PCA', 'UMAP') to apply to the data.")

    parser.add_argument('--n_components', type=int, default=30, 
                        help="Number of components for dimensionality reduction, if applicable.")

    args = parser.parse_args()
    return args


def decision_tree_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_dir', type=str, default='/home/syurtseven/GI-motility-quantification/segmentation/results/DecisionTree', 
                        help="Directory where the results and model files will be saved.")

    parser.add_argument('--exp_id', type=str, default='experiment/0', 
                        help="Identifier for the experiment, used to create subdirectories for results.")

    parser.add_argument('--criterion', type=str, default='gini', choices=['gini', 'entropy'], 
                        help="Function to measure the quality of a split at each node in the tree.")

    parser.add_argument('--splitter', type=str, default='best', choices=['best', 'random'], 
                        help="Strategy used to choose the split at each node.")

    parser.add_argument('--max_depth', type=int, default=None, 
                        help="The maximum depth of the tree. Limits the growth of the tree to avoid overfitting.")

    parser.add_argument('--min_samples_split', type=int, default=2, 
                        help="The minimum number of samples required to split an internal node.")

    parser.add_argument('--min_samples_leaf', type=int, default=1, 
                        help="The minimum number of samples required to be at a leaf node.")

    parser.add_argument('--seed', type=int, default=42, 
                        help="Random seed for reproducibility of results.")

    parser.add_argument('--mode', type=str, required=True, 
                        help="Operational mode of the script. Set to 'train' for training or 'test' for evaluation.")

    parser.add_argument('--n_samples', type=int, default=30, 
                        help="Number of samples to be loaded from the dataset.")

    parser.add_argument('--target_resize', type=int, nargs=2, default=(128, 128), 
                        help="Target dimensions for resizing images, specified as width and height (e.g., 256 256).")

    parser.add_argument('--scale_method', type=str, default='standard', 
                        help="Scaling method for preprocessing the data (e.g., 'standard' for StandardScaler or 'minmax').")

    parser.add_argument('--dim_reduction_method', type=str, default=None, 
                        help="Optional dimensionality reduction technique (e.g., 'PCA') to apply to the data.")

    parser.add_argument('--n_components', type=int, default=None, 
                        help="Number of components for dimensionality reduction, if applicable.")

    args = parser.parse_args()
    return args

def knn_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_dir', type=str, default='/home/syurtseven/GI-motility-quantification/segmentation/results/KNN', 
                        help="Directory where the results and model files will be saved.")

    parser.add_argument('--exp_id', type=str, default='experiment/0', 
                        help="Identifier for the experiment, used to create subdirectories for results.")

    parser.add_argument('--n_neighbors', type=int, default=5, 
                        help="Number of neighbors to use by default for KNN queries.")

    parser.add_argument('--weights', type=str, default='uniform', choices=['uniform', 'distance'], 
                        help="Weight function used in prediction. 'uniform' uses equal weights, 'distance' weights by inverse distance.")

    parser.add_argument('--algorithm', type=str, default='auto', choices=['auto', 'ball_tree', 'kd_tree', 'brute'], 
                        help="Algorithm to use for nearest neighbors computation. 'auto' tries to choose the best option.")

    parser.add_argument('--p', type=int, default=2, 
                        help="Power parameter for the Minkowski metric. p=2 corresponds to Euclidean distance.")

    parser.add_argument('--seed', type=int, default=42, 
                        help="Random seed for reproducibility of results.")

    parser.add_argument('--mode', type=str, required=True, 
                        help="Operational mode of the script. Set to 'train' for training or 'test' for evaluation.")

    parser.add_argument('--n_samples', type=int, default=30, 
                        help="Number of samples to be loaded from the dataset.")

    parser.add_argument('--target_resize', type=int, nargs=2, default=(128, 128), 
                        help="Target dimensions for resizing images, specified as width and height (e.g., 256 256).")

    parser.add_argument('--scale_method', type=str, default='standard', 
                        help="Scaling method for preprocessing the data (e.g., 'standard' for StandardScaler or 'minmax').")

    parser.add_argument('--dim_reduction_method', type=str, default=None, 
                        help="Optional dimensionality reduction technique (e.g., 'PCA') to apply to the data.")

    parser.add_argument('--n_components', type=int, default=None, 
                        help="Number of components for dimensionality reduction, if applicable.")

    args = parser.parse_args()
    return args

def save_arguments_to_file(args):

    args_dict = vars(args)
    os.makedirs(os.path.join(args.save_dir, args.exp_id), exist_ok=True)
    file_path = os.path.join(args.save_dir, args.exp_id,"args.json")

    with open(file_path, 'w') as f:
        json.dump(args_dict, f, indent=4)