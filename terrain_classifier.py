import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import image_preprocessing
import hog 
from operator import itemgetter
from skimage.feature import hog as ski_hog
from sklearn import svm
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from sklearn.inspection import DecisionBoundaryDisplay
import random

def validate_hog(image_path):
    """Compare custom hog implementation with sci-kit hog.

    Prints shape of both produced feature vectors, as well as their L2 difference and percentage mismatch. Ignores order difference in feature vector entries.

    Args:
        image_path (string): Relative file path of an image, of which a patch is taken to evaluate.

    """

    # Select one image.
    images = image_preprocessing.unsegmented_image_patch_extraction(image_path)
    # Put image into list for my implementations, alone for sklearn's.
    image = []
    image.append(images[0])
    ski_image = images[0]

    # Compute both HOG
    ski_results = ski_hog(
        np.array(ski_image),
        orientations=9,
        pixels_per_cell=(8,8),
        cells_per_block=(2,2),
        block_norm='L2-Hys',
        feature_vector=True
    )
    my_results = hog.extract_feature_vectors(image)[0]

    print("My shape:", my_results.shape)
    print("Sk-image shape:", ski_results.shape)

    # Sort both in L2 normal to account for entry ordering differences
    diff = np.linalg.norm(np.sort(my_results) - np.sort(ski_results))
    print("L2 difference:", diff)
    print("Relative difference percentage:", diff/np.linalg.norm(my_results))


def main():

    ############ IMAGE PREPROCESSING ####################
    testing_percentage = 0.30
    ### Collect image-patches from images in a directory as training data:(X, Y)=(patch image array, terrain label array)
        #all = image_preprocessing.segmented_directory_patch_extraction("data\\RUGD_frames-with-annotations\\creek", "data\\RUGD_annotations\\creek")
        #creek_train, creek_test = image_preprocessing.segmented_directory_patch_extraction("data\\RUGD_frames-with-annotations\\creek", "data\\RUGD_annotations\\creek", testing_percentage)
        #village_train, village_test = image_preprocessing.segmented_directory_patch_extraction("data\\RUGD_frames-with-annotations\\village", "data\\RUGD_annotations\\village", testing_percentage)    
        #trail11 = image_preprocessing.segmented_directory_patch_extraction("data\\RUGD_frames-with-annotations\\trail-11", "data\\RUGD_annotations\\trail-11")
    trail9_train, trail9_test = image_preprocessing.segmented_directory_patch_extraction("data\\RUGD_frames-with-annotations\\trail-9", "data\\RUGD_annotations\\trail-9", testing_percentage)


    ### Collect image-patches from an image as training data:(X, Y)=(patch image array, terrain label array)
        #creek_snap = image_preprocessing.segmented_image_patch_extraction("data\\RUGD_frames-with-annotations\\creek\\creek_00001.png", "data\\RUGD_annotations\\creek\\creek_00001.png")
        #village_snap = image_preprocessing.segmented_image_patch_extraction("data\\RUGD_frames-with-annotations\\village\\village_00003.png", "data\\RUGD_annotations\\village\\village_00003.png")
            # has good blend of grass and dirt
        #trail11_snap = image_preprocessing.segmented_image_patch_extraction("data\\RUGD_frames-with-annotations\\trail-11\\trail-11_01056.png", "data\\RUGD_annotations\\trail-11\\trail-11_01056.png")

    ### Unsegmented single image collection
        #trail11_snap_unseg = image_preprocessing.unsegmented_image_patch_extraction("data\\RUGD_frames-with-annotations\\trail-11\\trail-11_01056.png")
        #creek_snap_unseg = image_preprocessing.unsegmented_image_patch_extraction("data\\RUGD_frames-with-annotations\\creek\\creek_00001.png")


    ### Experimentation
        #print()
        #print(creek_snap)
        #print()
        #print(creek_snap_unseg)

    print("Images processed")

    ########### TRAINING DATA SELECTION ##################
    # Seperates training and testing full images randomly, not image patches.

    # Randomly remove testing_percent images from the training set
    # training = village
    # testing_percent = 0.30
    # testing_size = int(len(training) * testing_percent)
    # testing = [training.pop(random.randrange(len(training))) for _ in range(testing_size)]

    training = trail9_train
    testing = trail9_test

    # Seperate images and labels
    training_images = list(map(itemgetter(0), training))
    training_labels = list(map(itemgetter(1), training))

    testing_images = list(map(itemgetter(0), testing))
    testing_labels = list(map(itemgetter(1), testing))

    ## Experimentation
    
        # print()
        # print(testing_images)
        # print()
        # print(testing_labels)
        # print()

    print("Training and Testing Seperated")

    ############# HOG ####################

    ### HOG: image -> np-array of histogram entries. X has shape (histograms)
    ### Dimension of each entry is 36 [(4 cells/block) * (9 bins/cell-histogram)]

    # Extract features
    block_size = 3
    cell_size = 8
    training_features = hog.extract_feature_vectors(training_images, block_size, cell_size)
    testing_features = hog.extract_feature_vectors(testing_images, block_size, cell_size)


    ### Experimentation
        #validate_hog("data\\RUGD_frames-with-annotations\\creek\\creek_00001.png")

    print("Features Extracted")

    ###############  BUILDING MODEL  ################

    #model = svm.LinearSVC(max_iter=10000)
    kernels = ['linear', 'poly', 'rbf']
    for k in kernels:
        model = svm.SVC(kernel=k)
        model.fit(training_features, training_labels)
        print("Training Complete")
        model.predict(testing_features)
        print(f"\\ \\ -- Model with {k} kernel --")
        print(f"\\ Num of support vectors: {len(model.support_vectors_)}")
        print(f"\\ Accuracy: {model.score(testing_features, testing_labels)}\n")

    #############   Terrain-Mapped Testing Image Reconstruction

    
    #image_size = {,,3}
    #reconstruct_from_patches_2d()




    ##############    PLOT MANY MODELS: TO DO IN FUTURE    ##################
    # C = 1.0     # SVM Regularization parameter
    # models = (
    # svm.SVC(kernel="linear", C=C),
    # svm.LinearSVC(C=C, max_iter=10000),
    # svm.SVC(kernel="rbf", gamma=0.7, C=C),
    # svm.SVC(kernel="poly", degree=3, gamma="auto", C=C),
    # )
    # models = (clf.fit(training_features, training_labels) for clf in models)

    # titles = (
    # "SVC with linear kernel",
    # "LinearSVC (linear kernel)",
    # "SVC with RBF kernel",
    # "SVC with polynomial (degree 3) kernel",
    # )

    # # Set-up 2x2 grid for plotting.
    # fig, sub = plt.subplots(2, 2)
    # plt.subplots_adjust(wspace=0.4, hspace=0.4)


    return

if __name__ == "__main__":
    main()

