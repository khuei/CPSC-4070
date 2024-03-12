import numpy as np

def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    for Xi in range(0, Hi):
        for Yi in range(0, Wi):
            for Xk in range(-(Hk // 2), Hk // 2 + 1):
                for Yk in range(-(Wk // 2), Wk // 2 + 1):
                    XCoord = Xi - Xk
                    YCoord = Yi - Yk

                    if(XCoord >= 0 and XCoord < Hi and YCoord >= 0 and YCoord < Wi):
                        out[Xi, Yi] += image[XCoord, YCoord] * kernel[Xk + Hk // 2,Yk + Wk // 2]

                    
    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = np.zeros((H + 2 * pad_height, W + 2 * pad_width))
    
    out[pad_height:pad_height+H, pad_width:pad_width+W] = image
            
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    image_padded = zero_pad(image, Hk // 2, Wk // 2)

    kernel_flipped = np.flip(kernel, (0, 1))

    for i in range(Hi):
        for j in range(Wi):
            out[i, j] = np.sum(image_padded[i:i+Hk, j:j+Wk] * kernel_flipped)

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g.    
    
    Hint: use the conv_fast function defined above.
    Hint: subtract the mean of g from g so that its mean becomes zero.
    Hint: you should look up useful numpy functions online for calculating the mean.    

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """
    
    Hf, Wf = f.shape
    Hg, Wg = g.shape

    out = None
    
    g_flipped = np.flip(g, (0, 1))
    g_flipped -= np.mean(g)
    out = conv_fast(f, g_flipped)

    return out