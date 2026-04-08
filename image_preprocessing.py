import numpy as np
from PIL import Image
from functools import total_ordering
import glob
import os



# Integer terrain labels
@total_ordering
class Terrain():
    def __init__(self, rgb, ID, count):
        self.rgb = rgb
        self.ID = ID
        self.count = count

    def __eq__(self, other):
        if isinstance(other, Terrain):
            return self.count == other.count
        return NotImplemented
    
    def __lt__(self, other):
        if isinstance(other, Terrain):
            return self.count < other.count
        return NotImplemented



def integer_tiling(W, H, target):
    """Return an ideal near-square tile size and its image % coverage when in use. 

    Ideal tile size is defined by optimization of min(deviation from W and H + deviation from W = H).

    Args:
        W (int): width of image to tile
        H (int): height of image to tile
        target(int): desired length of tile side

    Returns: 
        tuple: (tile width, tile height, image coverage with use).
    """
    best = None
    min_penalty = float('inf')

    for cols in range(1, W+1):
        if W % cols == 0:
            w = W // cols
            for rows in range(1, H + 1):
                h = H // rows

                penalty = abs(w - target) + abs(h - target) + abs(w - h)

                if penalty < min_penalty:
                    min_penalty = penalty
                    best = (
                        w,
                        h,
                        (1 - (max(W%w,1) * max(H%h,1)) / (H * W)) # percentage coverage of tiling
                    )
    return best


def segmented_directory_patch_extraction(image_directory, segmentation_directory):
    """Converts images into a list of per-full-image lists of labeled image-patches.

    Filters patches such that all are dominated by a monitored terrain type.

    Args:
        image_directory (string): Relative file path of directory with images.
        segmentation_directory (string): Relative file path of directory with pixel-annotated images.

    Returns:
        list(list((Image, int))): List of per-full-image lists of image and their corresponding labels.
    
    """

    # Collect files recursively and sort by name to correspond images with their segmentation map
    image_paths = sorted(glob.glob(os.path.join(image_directory, "**/*.png"), recursive=True))
    segmentation_paths = sorted(glob.glob(os.path.join(segmentation_directory, "**/*.png"), recursive=True))
    
    # Store each (patch-array image, label) pair into labeled_images
    labeled_images = []
    for image, seg in zip(image_paths, segmentation_paths):
        labeled_images.append(segmented_image_patch_extraction(image, seg))


    return labeled_images


def unsegmented_directory_patch_extraction(image_directory):
    """Converts images into a list per-full-image lists of 64x64 image-patches.

    Does not filter patches in any way.

    Args:
        image_directory (string): Relative file path of directory with images.

    Returns:
        list(list(Image)): List of per-full-image lists of images, where each is represented as tiling patches.
    
    """
    
    # Collect files recursively and sort by name for deterministic output
    image_paths = sorted(glob.glob(os.path.join(image_directory, "**/*.png"), recursive=True))
    
    # Store each patch-array image into images
    images = []
    for image in image_paths:
        images.append(unsegmented_image_patch_extraction(image))
    
    return images

def segmented_image_patch_extraction(image_path, segmentation_path):
    """Converts image into a list of labeled, tiling image-patches.

    Filters patches such that all are dominated by a monitored terrain type.

    Args:
        image_path (string): Relative file path of images.
        segmentation_path (string): Relative file path of pixel-annotated image.

    Returns:
        list(Image, int): List of patch-label pairs.
    
    """
    
    # Parse image
    im = Image.open(image_path)

    # Convert to greymap with standard Luma transform
    im = im.convert("L")

    # Normalize Gamma
    gamma = 2.2
    im = im.point(lambda i: pow((i/255), 1/gamma) * 255)

    # Parse segmentation map
    seg = Image.open(segmentation_path)

    # Check segmentation map
    #seg.show()

    # Check image
    #im.show()

    patch_size = 64
    stride = 32
    patches = []
    seg_patches = []
    for y_upper in range(0, im.height - patch_size + 1, stride):
        for x_left in range(0, im.width - patch_size + 1, stride):
            patch = im.crop((x_left, y_upper, x_left + patch_size, y_upper + patch_size))
            patches.append(patch)
            seg_patch = seg.crop((x_left, y_upper, x_left + patch_size, y_upper + patch_size))
            seg_patches.append(seg_patch)


    ### View a patch
    #i = 7
    #patches[i].show()
    #seg_patches[i].show()


    # Surveying the following terrain labels
    dirt = Terrain((108, 64, 20), 1, 0)
    grass = Terrain((0, 102, 0), 2, 0)
    asphalt = Terrain((64, 64, 64), 3, 0)
    gravel = Terrain((255, 128, 0), 4, 0)


    # Save patches that are dominated (>60%) by one valid (above) terrain types
    dominated_patches = []
    threshold = 0.6
    for patch, seg_patch in zip(patches, seg_patches):

        # Convert patch into np array
        patch_array = np.array(seg_patch)

        # Apply terrain masks and count remaining non-zero values for each
        dirt.count    = np.count_nonzero(np.all(patch_array == dirt.rgb, axis=-1))
        grass.count   = np.count_nonzero(np.all(patch_array == grass.rgb, axis=-1))
        asphalt.count = np.count_nonzero(np.all(patch_array == asphalt.rgb, axis=-1))
        gravel.count  = np.count_nonzero(np.all(patch_array == gravel.rgb, axis=-1))
        
        # Check terrain spread
        #print(f"Dirt: {dirt.count}, Grass: {grass.count}, asphalt: {asphalt.count}, gravel: {gravel.count}")

        largest_terrain = max(dirt, grass, asphalt, gravel)
        if (largest_terrain.count/(patch_size * patch_size) >= threshold):
            dominated_patches.append((patch, largest_terrain.ID))


    ### View all dominatedpatches
    #for img in dominated_patches: 
    #    img[0].show()

    return dominated_patches



def unsegmented_image_patch_extraction(image):
    """Converts image into a list of tiling image-patches. 
    
    Does not filter patches in any way.

    Args:
        image_path (string): Relative file path of images.

    Returns:
        list(Image): List of patches.
    
    """

    # Parse image
    im = Image.open(image)

    # Convert to greymap with standard Luma transform
    im = im.convert("L")

    # Normalize Gamma
    gamma = 2.2
    im = im.point(lambda i: pow((i/255), 1/gamma) * 255)

    patch_size = 64
    stride = 32
    patches = []
    for y_upper in range(0, im.height - patch_size + 1, stride):
        for x_left in range(0, im.width - patch_size + 1, stride):
            patch = im.crop((x_left, y_upper, x_left + patch_size, y_upper + patch_size))
            patches.append(patch)

    return patches

