import numpy as np
from scipy.signal import convolve2d
from scipy.linalg import norm as l2_norm

def l2_hys_norm(v, threshold=0.2):
    # Find L2 norm
    norm = l2_norm(v)
    if norm > 0:
        v = v/norm
    # Apply hystersis (bound to interval(0, threshold))
    v = np.clip(v, 0, threshold)

    # Re-normalize
    norm =  l2_norm(v)
    if norm > 0:
        v = v/norm

    return v

def extract_feature_vectors(images):

    # Define gradient kernels
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1] 
        ])
    
    sobel_y = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1] 
        ])
    
    # features = (images, cells, hog)

    bins = 9
    bin_size = 180/bins
    # Dimension = blocks * cells/block * histogram-entries/cell
    D = 49 * 4 * bins
    features = np.zeros((len(images), 49*4*bins), dtype=float)

    for img_idx, image in enumerate(images):

        # Convert image to np array of floats
        image = np.array(image).astype(np.float32) / 255.0
        
        # Add symmetric padding
        image = np.pad(image, 1, mode='reflect')

        # Compute Gradients
        G_x = convolve2d(image, sobel_x, mode='valid')
        G_y = convolve2d(image, sobel_y, mode='valid')

        mag = np.hypot(G_x, G_y)
        ori = np.atan2(G_y, G_x)

        # Split image into 8x8 cells
        # "Tiles" the array in view only, no memory copying. 8x8 grid of 8x8 cells.
        mag_cells = mag.reshape(8, 8, 8, 8).transpose(0, 2, 1, 3)
        ori_cells = ori.reshape(8, 8, 8, 8).transpose(0, 2, 1, 3)

        # Flattens the view grid into sixty-four 8x8 cells
        mag_cells = mag_cells.reshape(64, 8, 8)
        ori_cells = ori_cells.reshape(64, 8, 8)

        # Compute histograms for each cell
        cell_histograms = np.zeros((64,bins), dtype=float)
        for cell_idx, (mag_cell, ori_cell) in enumerate(zip(mag_cells, ori_cells)):
            hog = np.zeros(bins)
            for mag, ori in zip(mag_cell.ravel(), ori_cell.ravel()):
                # Map [0,2pi]->[0,180)s
                theta = (np.degrees(ori) % 180)
                # Linearly interpolate magnitude between nearest bins
                bin_float = theta / bin_size
                lower_bin = int(bin_float) % bins
                upper_bin = (lower_bin + 1) % bins

                hog[upper_bin] += mag * (bin_float - lower_bin)
                hog[lower_bin] += mag * (upper_bin - bin_float)

            # Add cell histogram to larger array  
            cell_histograms[cell_idx] = hog

        # Reshape cell-histograms into grid  -> (cell_x, cell_y, histogram) 
        cell_histograms = cell_histograms.reshape(8,8,bins)

        block_histograms = []
        # Apply L2-Hys norm to each block
        for i in range(7):
            for j in range(7):
                block = cell_histograms[i:i+2, j:j+2].flatten()
                normalized_block = l2_hys_norm(block)
                block_histograms.extend(normalized_block)
                                        
        features[img_idx] = np.array(block_histograms)

    return features