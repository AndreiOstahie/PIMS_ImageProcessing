from PIL import Image
import numpy as np
import cv2


def median_filter2(image, ksize=5):
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


def non_local_means_denoising(image, h):
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


def bilateral_filter_denoising(image, diameter=9, sigma_color=75, sigma_space=75):
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
