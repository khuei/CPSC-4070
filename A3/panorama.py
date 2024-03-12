import numpy as np
from skimage import filters
from skimage.feature import corner_peaks
from skimage.util.shape import view_as_blocks
from scipy.spatial.distance import cdist
from scipy.ndimage.filters import convolve

from utils import pad, unpad, get_output_space, warp_image


def harris_corners(img, window_size=3, k=0.04):
    """
    Compute Harris corner response map. Follow the math equation
    R=Det(M)-k(Trace(M)^2).

    Hint:
        You may use the function scipy.ndimage.filters.convolve,
        which is already imported above.

    Args:
        img: Grayscale image of shape (H, W)
        window_size: size of the window function
        k: sensitivity parameter

    Returns:
        response: Harris response image of shape (H, W)
    """

    H, W = img.shape
    window = np.ones((window_size, window_size))

    response = np.zeros((H, W))

    dx = filters.sobel_v(img)
    dy = filters.sobel_h(img)

    dx2, dy2, dxdy = dx ** 2, dy ** 2, dx * dy
    mx2 = convolve(dx2, window, mode='constant')
    my2 = convolve(dy2, window, mode='constant')
    mxy = convolve(dxdy, window, mode='constant')
    response = (mx2 * my2 - mxy ** 2) - k * (mx2 + my2) ** 2

    return response



def simple_descriptor(patch):
    """
    Describe the patch by normalizing the image values into a standard
    normal distribution (having mean of 0 and standard deviation of 1)
    and then flattening into a 1D array.

    The normalization will make the descriptor more robust to change
    in lighting condition.

    Hint:
        If a denominator is zero, divide by 1 instead.

    Args:
        patch: grayscale image patch of shape (H, W)

    Returns:
        feature: 1D array of shape (H * W)
    """

    mean = np.mean(patch)
    std = np.std(patch)

    if std == 0:
        std = 1

    feature = ((patch - mean) / std).reshape(-1)

    return feature


def describe_keypoints(image, keypoints, desc_func, patch_size=16):
    """
    Args:
        image: grayscale image of shape (H, W)
        keypoints: 2D array containing a keypoint (y, x) in each row
        desc_func: function that takes in an image patch and outputs
            a 1D feature vector describing the patch
        patch_size: size of a square patch at each keypoint

    Returns:
        desc: array of features describing the keypoints
    """

    image.astype(np.float32)
    desc = []

    for i, kp in enumerate(keypoints):
        y, x = kp
        patch = image[y-(patch_size//2):y+((patch_size+1)//2),
                      x-(patch_size//2):x+((patch_size+1)//2)]
        desc.append(desc_func(patch))

    return np.array(desc)


def match_descriptors(desc1, desc2, threshold=0.5):
    """
    Match the feature descriptors by finding distances between them. A match is formed
    when the distance to the closest vector is much smaller than the distance to the
    second-closest, that is, the ratio of the distances should be smaller
    than the threshold. Return the matches as pairs of vector indices.

    Hint:
        The Numpy functions np.sort, np.argmin, np.asarray might be useful

    Args:
        desc1: an array of shape (M, P) holding descriptors of size P about M keypoints
        desc2: an array of shape (N, P) holding descriptors of size P about N keypoints

    Returns:
        matches: an array of shape (Q, 2) where each row holds the indices of one pair
        of matching descriptors
    """
    matches = []

    N = desc1.shape[0]
    dists = cdist(desc1, desc2)

    matches = np.concatenate([np.arange(N).reshape(-1, 1), np.argmin(dists, axis=1).reshape(-1, 1)], axis=1)
    d = np.sort(dists, axis=1)
    matches = matches[d[:, 0] / d[:, 1] < threshold]

    return matches

def ransac(keypoints1, keypoints2, matches, n_iters=200, threshold=20):
    """
    Use RANSAC to find a robust affine transformation

        1. Select random set of matches
        2. Compute affine transformation matrix
        3. Compute inliers
        4. Keep the largest set of inliers
        5. Re-compute least-squares estimate on all of the inliers

    Args:
        keypoints1: M1 x 2 matrix, each row is a point
        keypoints2: M2 x 2 matrix, each row is a point
        matches: N x 2 matrix, each row represents a match
            [index of keypoint1, index of keypoint 2]
        n_iters: the number of iterations RANSAC will run
        threshold: the number of threshold to find inliers

    Returns:
        H: a robust estimation of affine transformation from keypoints2 to
        keypoints 1
    """
    # Copy matches array, to avoid overwriting it
    orig_matches = matches.copy()
    matches = matches.copy()

    N = matches.shape[0]
    print(N)
    n_samples = int(N * 0.2)

    matched1 = pad(keypoints1[matches[:,0]])
    matched2 = pad(keypoints2[matches[:,1]])

    max_inliers = np.zeros(N)
    n_inliers = 0

    # RANSAC iteration start
    for _ in range(n_iters):
        indices = np.random.choice(N, n_samples, replace=False)
        s1, s2 = matched1[indices], matched2[indices]
        H = np.linalg.lstsq(s2, s1, rcond=None)[0]
        H[:,2] = np.array([0, 0, 1])
        curr_inliers = np.sqrt(((matched2.dot(H) - matched1) ** 2).sum(axis=-1)) < threshold
        curr_n_inliers = curr_inliers.sum()
        if curr_n_inliers > n_inliers:
            max_inliers, n_inliers = curr_inliers, curr_n_inliers
    H = np.linalg.lstsq(matched2[max_inliers], matched1[max_inliers], rcond=None)[0]
    H[:,2] = np.array([0, 0, 1])
    print(H)
    return H, orig_matches[max_inliers]


def hog_descriptor(patch, pixels_per_cell=(8,8)):
    """
    Ref: http://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf
    
    Generating hog descriptor by the following steps:

    1. Compute the gradient image in x and y directions (already done for you)
    2. Compute gradient histograms for each cell
    3. Flatten block of histograms into a 1D feature vector
        Here, we treat the entire patch of histograms as our block
    4. Normalize flattened block
        Normalization makes the descriptor more robust to lighting variations

    Args:
        patch: grayscale image patch of shape (H, W)
        pixels_per_cell: size of a cell with shape (M, N)

    Returns:
        block: 1D patch descriptor array of shape ((H*W*n_bins)/(M*N))
    """
    assert (patch.shape[0] % pixels_per_cell[0] == 0),\
                'Heights of patch and cell do not match'
    assert (patch.shape[1] % pixels_per_cell[1] == 0),\
                'Widths of patch and cell do not match'

    n_bins = 9
    degrees_per_bin = 180 // n_bins

    Gx = filters.sobel_v(patch)
    Gy = filters.sobel_h(patch)

    # Unsigned gradients
    G = np.sqrt(Gx**2 + Gy**2)
    theta = (np.arctan2(Gy, Gx) * 180 / np.pi) % 180

    # Group entries of G and theta into cells of shape pixels_per_cell, (M, N)
    #   G_cells.shape = theta_cells.shape = (H//M, W//N)
    #   G_cells[0, 0].shape = theta_cells[0, 0].shape = (M, N)
    G_cells = view_as_blocks(G, block_shape=pixels_per_cell)
    theta_cells = view_as_blocks(theta, block_shape=pixels_per_cell)
    rows = G_cells.shape[0]
    cols = G_cells.shape[1]

    # For each cell, keep track of gradient histrogram of size n_bins
    cells = np.zeros((rows, cols, n_bins))

    # Compute histogram per cell
    lower_bins = ((theta_cells // degrees_per_bin - (theta_cells % degrees_per_bin <= degrees_per_bin / 2)) % n_bins).astype(int)
    upper_bins = ((lower_bins + 1) % n_bins).astype(int)

    lower_weights = ((degrees_per_bin / 2 - theta_cells % degrees_per_bin) % degrees_per_bin) / degrees_per_bin
    upper_weights = 1 - lower_weights

    indices = np.indices((rows, cols, pixels_per_cell[0], pixels_per_cell[1]))
    i, j = indices[0].reshape(-1), indices[1].reshape(-1)

    np.add.at(cells, (i, j, lower_bins.reshape(-1)), (lower_weights * G_cells).reshape(-1))
    np.add.at(cells, (i, j, upper_bins.reshape(-1)), (upper_weights * G_cells).reshape(-1))

    block = cells.reshape(-1)
    block /= np.linalg.norm(block)

    return block
