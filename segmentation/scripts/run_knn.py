import os
import pickle
import glob
import sys
sys.path.append(os.path.dirname(os.getcwd()))

from sklearn.neighbors import KNeighborsClassifier

from src.utils.preprocessing import load_images_from_folder
from src.utils.args_utils import knn_arg_parser, save_arguments_to_file  # Modify arg parser for KNN
from src.utils.model_utils import *
from src.utils.variable_utils import MADISON_STOMACH
from src.utils.metric_utils import compute_dice_score_np
from src.utils.viz_utils import visualize_predictions2



if __name__ == '__main__':
    # Parse arguments for K-Nearest Neighbors (KNN) model
    args = knn_arg_parser()  # Modified argument parser for KNN

    # Define the save directory and experiment ID
    save_path = os.path.join(args.save_dir, args.exp_id)

    # Save the arguments to a file for reproducibility
    save_arguments_to_file(args=args)

    # Set the random seed for reproducibility
    set_seed(args.seed)

    # Load images and masks from the dataset with specified preprocessing
    images, masks = load_images_from_folder(dataset=MADISON_STOMACH,
                                            mode=args.mode, 
                                            n_samples=args.n_samples,
                                            target_resize=args.target_resize,
                                            scaler=args.scale_method,
                                            dim_reduction_method=args.dim_reduction_method,
                                            n_components=args.n_components)

    # Print shapes of the loaded data for verification
    print(f"Images shape {images.shape}")
    print(f"Masks shape {masks.shape}")

    # Reshape images and masks for compatibility with KNN
    images_reshaped = images.reshape(-1, 1)
    masks_reshaped = masks.reshape(-1)

    # Training mode
    if args.mode == 'train':
        print("Training starts")
        # Initialize the KNN classifier with hyperparameters from args
        clf_knn = KNeighborsClassifier(n_neighbors=args.n_neighbors, 
                                       weights=args.weights, 
                                       algorithm=args.algorithm, 
                                       p=args.p)
        # Train the KNN model
        clf_knn.fit(images_reshaped, masks_reshaped)
        
        # Save the trained KNN model to a file
        filename = os.path.join(save_path, f'knn_segmentation_model_{args.n_samples}.sav')
        pickle.dump(clf_knn, open(filename, 'wb'))

    # Testing mode
    elif args.mode == 'test':
        print("Testing starts")
        # Load the saved KNN model from the file
        with open(glob.glob(os.path.join(save_path, f'knn_segmentation_model_*.sav'))[0], 'rb') as file:
            clf_knn = pickle.load(file)
    
    # Make predictions on the reshaped images
    preds = clf_knn.predict(images_reshaped)
    # Reshape predictions back to the original mask dimensions
    preds = preds.reshape(masks.shape)

    # Compute the Dice score to evaluate the segmentation performance
    dice_score = compute_dice_score_np(preds, masks)
    dice_message = f"Dice Score on {args.mode} data: {dice_score:.4f}"
    print(dice_message)

    # Visualize predictions for a subset of the dataset
    for i in range(5):
        visualize_predictions2(images=images[i*5:5*(i+1), :, :], 
                               masks=masks[i*5:5*(i+1), :, :], 
                               outputs=preds[i*5:5*(i+1), :, :], 
                               save_path=save_path, 
                               batch_idx=i)

    # Save the Dice score to a file
    dice_score_filename = os.path.join(save_path, f'dice_score_{args.mode}.txt')
    with open(dice_score_filename, 'w') as file:
        file.write(dice_message)
