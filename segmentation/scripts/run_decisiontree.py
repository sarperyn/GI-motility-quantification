import os
import pickle
import glob
import sys
sys.path.append(os.path.dirname(os.getcwd()))

from sklearn.tree import DecisionTreeClassifier

from src.utils.preprocessing import load_images_from_folder
from src.utils.args_utils import decision_tree_arg_parser, save_arguments_to_file
from src.utils.model_utils import *
from src.utils.variable_utils import MADISON_STOMACH
from src.utils.metric_utils import compute_dice_score_np
from src.utils.viz_utils import visualize_predictions2

if __name__ == '__main__':
    
    args = decision_tree_arg_parser()
    save_path = os.path.join(args.save_dir, args.exp_id)
    save_arguments_to_file(args=args)

    set_seed(args.seed)

    images, masks = load_images_from_folder(dataset=MADISON_STOMACH,
                                            mode=args.mode, 
                                            n_samples=args.n_samples,
                                            target_resize=args.target_resize,
                                            scaler=args.scale_method,
                                            dim_reduction_method=args.dim_reduction_method,
                                            n_components=args.n_components)

    print(f"Images shape {images.shape}")
    print(f"Masks shape {masks.shape}")

    images_reshaped = images.reshape(-1, 1)
    masks_reshaped = masks.reshape(-1)

    if args.mode == 'train':
        print("Training starts")
        clf_tree = DecisionTreeClassifier(criterion=args.criterion,
                                          splitter=args.splitter,
                                          max_depth=args.max_depth,
                                          min_samples_split=args.min_samples_split,
                                          min_samples_leaf=args.min_samples_leaf,
                                          random_state=args.seed)
        clf_tree.fit(images_reshaped, masks_reshaped)
        
        filename = os.path.join(save_path, f'decision_tree_segmentation_model_{args.n_samples}.sav')
        pickle.dump(clf_tree, open(filename, 'wb'))

    elif args.mode == 'test':
        print("Testing starts")
        with open(glob.glob(os.path.join(save_path, f'decision_tree_segmentation_model_*.sav'))[0], 'rb') as file:
            clf_tree = pickle.load(file)
    
    preds = clf_tree.predict(images_reshaped)
    preds = preds.reshape(masks.shape) 

    dice_score = compute_dice_score_np(preds, masks)
    dice_message = f"Dice Score on {args.mode} data: {dice_score:.4f}"
    print(dice_message)

    for i in range(5):
        visualize_predictions2(images=images[i*5:5*(i+1),:,:], 
                          masks=masks[i*5:5*(i+1),:,:], 
                          outputs=preds[i*5:5*(i+1),:,:], 
                          save_path=save_path, 
                          batch_idx=i)

    dice_score_filename = os.path.join(save_path, f'dice_score_{args.mode}.txt')
    with open(dice_score_filename, 'w') as file:
        file.write(dice_message)
