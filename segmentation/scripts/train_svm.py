import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))

from src.models.svm import load_images_from_folder, get_configs
from src.utils.model_utils import train, predict


if __name__ == '__main__':

    config_path = '/home/syurtseven/GI-motility-quantification/segmentation/config/experiments/MadisonSVM.yaml'
    X_train, y_train = load_images_from_folder(mode='train')
    print(X_train.shape)
    print(y_train.shape)

    svm_model, kernel, scaler, dim_reduction, save_path = get_configs(config_path)

    svm_clf = svm_model(kernel=kernel)

    print("SVM Training started")
    model = train(model=svm_clf,
                X_train=X_train,
                y_train=y_train)
    
    X_test, y_test = load_images_from_folder(mode='test')
    y_preds = model.predict(X_test)

    print(y_preds.shape)