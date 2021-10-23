import torch
import numpy as np
from torchvision.transforms import ToTensor
from sklearn import model_selection

def get_imgs_mean_stddev(imgs, axis=None):    
    """Get the mean and standard deviation for images in a dataset / mini-batch
    Args:
        imgs ([2d or 3d numpy array]): images in collection (with no to_tensor transformation applied)
        axis ([tuple of ints], optional): Axis along which mean and std dev is to be calculated.
        Defaults to None.
    Returns:
        [tuple]: tuple of tensors with mean and std.dev. of the imgs
    """
    to_tensor = ToTensor()
    img_tensor_arr = [to_tensor(img) for img in imgs]
    # stack will arrange the tensors one over the other with dim=0 being the new dimension that  
    # stores the number of tensors stacked. This new dimension can be placed at any index
    img_tensor_arr = torch.stack(img_tensor_arr)
    if axis is not None:
        return torch.mean(img_tensor_arr, axis=axis), torch.std(img_tensor_arr, axis=axis)
    else:            
        return torch.mean(img_tensor_arr, axis=(0, 2, 3)), torch.std(img_tensor_arr, axis=(0,2,3))

# for a training and label data in form of numpy arrays, return a fold_index array whose elements
# represent the fold index. The length of this fold_index array is same as length of input dataset
# and the unique fold values represent the items to be used for validation in the corresponding
# cross validation iteration with rest of the items being used for training (typical ration being 80:20)
def get_skf_index(num_folds, X, y):
    skf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state = 42)
    train_fold_index = np.zeros(len(y))
    for fold, (train_index, val_index) in enumerate(skf.split(X=X, y=y)):
        train_fold_index[val_index] = [fold + 1] * len(val_index)
    return train_fold_index        

# split the training dataframe into kfolds for cross validation. We do this before any processing is done
# on the data. We use stratified kfold if the target distribution is unbalanced
def strat_kfold_dataframe(df, target_col_name, num_folds=5):
    # we create a new column called kfold and fill it with -1
    df["kfold"] = -1
    # randomize of shuffle the rows of dataframe before splitting is done
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    # get the target data
    y = df["target"].values
    skf = model_selection.StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    for fold, (train_index, val_index) in enumerate(skf.split(X=df, y=y)):
        df.loc[val_index, "kfold"] = fold    
    return df        