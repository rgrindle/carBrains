# carBrains

To use this code:
1. Download and extract these files (they should be all in the same directory):
    http://imagenet.stanford.edu/internal/car196/cars_train.tgz
    http://imagenet.stanford.edu/internal/car196/cars_test.tgz
    https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz
2. Their testing data does not include labels, so also download http://imagenet.stanford.edu/internal/car196/cars_test_annos_withlabels.mat, rename it to cars_test_annos.mat and replace the existing one in the devkit with this new one with labels.
3. Add a new environment variable called CARS_DATASET_PATH, which links to the directory you saved the 3 stanford directories
