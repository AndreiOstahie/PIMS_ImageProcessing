from PIL import Image
import numpy as np
import cv2


# ========= Denoising methods using pre-built library functions (e.g. OpenCV - cv2.medianBlur) =========
def median_filter(image, ksize=5):
    """
    Apply median filter to an image
    :param image: PIL Image object
    :param ksize: Kernel/Window size (odd number)
    :return denoised_image: Denoised PIL Image object
    """

    # Convert image to numpy array (pixels)
    image_array = np.array(image)

    # Convert RGB to BGR for OpenCV processing
    image_array_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    # Apply the median filter
    denoised_array_bgr = cv2.medianBlur(image_array_bgr, ksize)

    # Convert the denoised image back to RGB
    denoised_array_rgb = cv2.cvtColor(denoised_array_bgr, cv2.COLOR_BGR2RGB)

    # Convert numpy array back to PIL image
    denoised_image = Image.fromarray(denoised_array_rgb)

    return denoised_image


def gaussian_filter(image, ksize=5):
    """
    Apply Gaussian filter to an image
    :param image: PIL Image object
    :return denoised_image: Denoised PIL Image object
    """

    # Convert image to numpy array (pixels)
    image_array = np.array(image)

    # Convert RGB to BGR for OpenCV processing
    image_array_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    # Apply Gaussian filter
    denoised_array_bgr = cv2.GaussianBlur(image_array_bgr, (ksize, ksize), 0)

    # Convert the denoised image back to RGB
    denoised_array_rgb = cv2.cvtColor(denoised_array_bgr, cv2.COLOR_BGR2RGB)

    # Convert numpy array back to PIL image
    denoised_image = Image.fromarray(denoised_array_rgb)

    return denoised_image


def non_local_means(image, h):
    """
    Apply Non-local Means denoising to an image
    :param: image: PIL Image object
    :return: denoised_image: Denoised PIL Image object
    """

    # Convert image to numpy array (pixels)
    image_array = np.array(image)

    # Convert RGB to BGR for OpenCV processing
    image_array_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    # Apply Non-local Means denoising
    denoised_array_bgr = cv2.fastNlMeansDenoisingColored(image_array_bgr, None, h, h, 7, 21)

    # Convert the denoised image back to RGB
    denoised_array_rgb = cv2.cvtColor(denoised_array_bgr, cv2.COLOR_BGR2RGB)

    # Convert numpy array back to PIL image
    denoised_image = Image.fromarray(denoised_array_rgb)

    return denoised_image


def bilateral_filter(image, diameter=9, sigma_color=75, sigma_space=75):
    """
    Apply bilateral filter to an image
    :param image: PIL Image object
    :param diameter: Diameter of each pixel neighborhood used during filtering
    :param sigma_color: Filter sigma in color space
    :param sigma_space: Filter sigma in coordinate space
    :return denoised_image: Denoised PIL Image object
    """

    # Convert image to numpy array (pixels)
    image_array = np.array(image)

    # Check if the image is grayscale or color
    if len(image_array.shape) == 2:  # Grayscale image
        denoised_image_array = cv2.bilateralFilter(image_array, diameter, sigma_color, sigma_space)
    else:  # Color image
        # Convert RGB to BGR for OpenCV processing
        image_array_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        denoised_image_array_bgr = cv2.bilateralFilter(image_array_bgr, diameter, sigma_color, sigma_space)

        # Convert the denoised image back to RGB
        denoised_image_array = cv2.cvtColor(denoised_image_array_bgr, cv2.COLOR_BGR2RGB)

    # Convert numpy array back to PIL image
    denoised_image = Image.fromarray(denoised_image_array)

    return denoised_image




# ========= Custom denoising methods =========
def median_filter_2(image, ksize=5):
    """
    Apply median filter to an image
    :param image: PIL Image object
    :param ksize: Kernel/Window size (odd number)
    :return denoised_image: Denoised PIL Image object
    """

    # Convert image to numpy array (pixels)
    img_array = np.array(image)

    # Pad the image array for boundary processing
    pad_size = ksize // 2

    if img_array.ndim == 2:  # Grayscale image
        padded_img = np.pad(img_array, pad_size, mode='edge')

        # Array for denoised image
        denoised_array = np.zeros_like(img_array)

        # Move the kernel across the image
        for i in range(img_array.shape[0]):
            for j in range(img_array.shape[1]):
                # Extract the neighborhood
                window = padded_img[i:i + ksize, j:j + ksize]

                # Calculate the median and set it to the corresponding pixel
                denoised_array[i, j] = np.median(window)
    else:  # Color image
        padded_img = np.pad(img_array, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='edge')

        # Array for denoised image
        denoised_array = np.zeros_like(img_array)

        # Process each channel separately
        for channel in range(img_array.shape[2]):
            # Move the kernel across the image
            for i in range(img_array.shape[0]):
                for j in range(img_array.shape[1]):
                    # Extract the neighborhood
                    window = padded_img[i:i + ksize, j:j + ksize, channel]

                    # Calculate the median and set it to the corresponding pixel
                    denoised_array[i, j, channel] = np.median(window)

    # Convert numpy array back to PIL image
    denoised_image = Image.fromarray(denoised_array)

    return denoised_image


def gaussian_kernel(size, sigma=1.0):
    """ Returns a 2D Gaussian kernel array """
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(size-1)/2)**2+(y-(size-1)/2)**2)/(2*sigma**2)), (size, size))
    return kernel / np.sum(kernel)

def gaussian_filter_2(image, ksize=5, sigma=1.0):
    """
    Apply Gaussian filter to an image using a manually defined Gaussian kernel
    :param image: PIL Image object
    :param ksize: Size of the kernel (odd number)
    :param sigma: Standard deviation of the Gaussian kernel
    :return: Denoised PIL Image object
    """

    if ksize % 2 == 0:
        raise ValueError("Kernel size must be odd")

    # Convert image to numpy array
    image_array = np.array(image)

    # Create a Gaussian kernel
    kernel = gaussian_kernel(ksize, sigma)

    # Pad the image to handle borders
    pad_size = ksize // 2

    if image_array.ndim == 2:  # Grayscale image
        padded_image = np.pad(image_array, pad_size, mode='edge')
        height, width = image_array.shape
        channels = 1
    else:  # Color image
        padded_image = np.pad(image_array, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='reflect')
        height, width, channels = image_array.shape

    # Prepare an output array
    denoised_array = np.zeros_like(image)

    # Apply the Gaussian kernel
    for y in range(height):
        for x in range(width):
            if channels == 1:  # Grayscale image
                denoised_array[y, x] = np.sum(kernel * padded_image[y:y + ksize, x:x + ksize])
            else:  # Color image
                for c in range(channels):
                    denoised_array[y, x, c] = np.sum(kernel * padded_image[y:y + ksize, x:x + ksize, c])

    # Convert the denoised numpy array back to a PIL image
    denoised_image = Image.fromarray(denoised_array.astype('uint8'))

    return denoised_image


def gaussian_distance(patch1, patch2, h):
    """Compute the Gaussian distance between two patches"""
    diff = patch1 - patch2
    norm = np.sum(diff**2)
    return np.exp(-norm / (h**2))


def non_local_means_2(image, h=10, patch_size=7, search_window=21):
    """
    Apply Non-local Means denoising algorithm to an image manually
    :param image: PIL Image object
    :param h: Filtering strength
    :param patch_size: Size of the patch used in the NLMD algorithm
    :param search_window: Size of the window used to search for patches
    :return: Denoised PIL Image object
    """

    img_array = np.array(image, dtype=float)

    if img_array.ndim == 2:  # Grayscale image
        height, width = img_array.shape
        channels = 1
        img_array = img_array[:, :, np.newaxis]  # Add a dummy channel dimension
    else:  # Color image
        height, width, channels = img_array.shape

    padded_image = np.pad(img_array, [(patch_size // 2, patch_size // 2), (patch_size // 2, patch_size // 2), (0, 0)], mode='reflect')
    denoised_image = np.zeros((height, width, channels))

    for y in range(height):
        for x in range(width):
            i_min = max(y - search_window // 2, 0)
            i_max = min(y + search_window // 2 + 1, height)
            j_min = max(x - search_window // 2, 0)
            j_max = min(x + search_window // 2 + 1, width)

            ref_patch = padded_image[y:y + patch_size, x:x + patch_size, :]

            weights = np.zeros((i_max - i_min, j_max - j_min))
            norm_factor = 0

            for i in range(i_min, i_max):
                for j in range(j_min, j_max):
                    patch = padded_image[i:i + patch_size, j:j + patch_size, :]
                    weight = gaussian_distance(ref_patch, patch, h)
                    weights[i - i_min, j - j_min] = weight
                    norm_factor += weight

            average = np.zeros(channels)
            for i in range(i_min, i_max):
                for j in range(j_min, j_max):
                    weight = weights[i - i_min, j - j_min]
                    average += weight * img_array[i, j, :]

            denoised_image[y, x, :] = average / norm_factor

    if channels == 1:  # Grayscale image
        denoised_image = denoised_image[:, :, 0]  # Remove the dummy channel dimension

    return Image.fromarray(np.uint8(denoised_image))


def gaussian(x, sigma):
    return np.exp(-(x ** 2) / (2 * sigma ** 2))


def bilateral_filter_2(image, diameter=9, sigma_color=75, sigma_space=75):
    """
    Manually apply bilateral filter to an image
    :param image: PIL Image object
    :param diameter: Diameter of each pixel neighborhood used during filtering
    :param sigma_color: Filter sigma in the color space
    :param sigma_space: Filter sigma in the coordinate space
    :return: Denoised PIL Image object
    """

    # Convert image to numpy array
    img_array = np.array(image)
    if img_array.ndim == 2:  # Grayscale image
        img_array = img_array[:, :, np.newaxis]  # Add a dummy channel dimension

    # Get dimensions
    height, width, channels = img_array.shape

    # Prepare output array
    filtered_image = np.zeros_like(img_array)

    # Define the half window
    half = diameter // 2

    # Iterate over each pixel
    for y in range(height):
        for x in range(width):
            for c in range(channels):
                w_p = 0
                i_filtered = 0
                for k in range(-half, half + 1):
                    for l in range(-half, half + 1):
                        ny = y + k
                        nx = x + l
                        if 0 <= ny < height and 0 <= nx < width:
                            g_s = gaussian(np.sqrt(k ** 2 + l ** 2), sigma_space)
                            g_r = gaussian(abs(int(img_array[ny, nx, c]) - int(img_array[y, x, c])), sigma_color)
                            w = g_s * g_r
                            i_filtered += img_array[ny, nx, c] * w
                            w_p += w
                filtered_image[y, x, c] = i_filtered / w_p

    # Handle the case of a grayscale image
    if filtered_image.shape[2] == 1:
        filtered_image = filtered_image.reshape(filtered_image.shape[0], filtered_image.shape[1])

    # Convert numpy array back to PIL image
    return Image.fromarray(np.uint8(filtered_image))


