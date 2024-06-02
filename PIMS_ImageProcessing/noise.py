from PIL import Image
import numpy as np
import cv2


def add_gaussian_noise(image, mean=0, std=25):
    """
    Apply Gaussian noise to an image
    :param image: PIL Image object
    :param mean: Mean of the Gaussian noise
    :param std: Standard deviation of the Gaussian noise
    :return: Noisy image as a PIL Image object
    """

    # Convert image to numpy array (pixels)
    image_array = np.array(image)

    # Generate Gaussian noise
    noise = np.random.normal(mean, std, image_array.shape)

    # Add noise to the image
    noisy_image_array = image_array + noise

    # Image boundaries
    noisy_image_array = np.clip(noisy_image_array, 0, 255)

    # Convert numpy array back to PIL image
    noisy_image = Image.fromarray(noisy_image_array.astype('uint8'))

    return noisy_image


def add_salt_and_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    """
    Apply Salt and Pepper noise to an image
    :param image: PIL Image object
    :param salt_prob: Probability of adding salt noise
    :param pepper_prob: Probability of adding pepper noise
    :return: Noisy image as a PIL Image object
    """

    # Convert image to numpy array (pixels)
    image_array = np.array(image)

    # Salt noise
    num_salt = np.ceil(salt_prob * image_array.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image_array.shape]
    image_array[tuple(coords)] = 255

    # Pepper noise
    num_pepper = np.ceil(pepper_prob * image_array.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image_array.shape]
    image_array[tuple(coords)] = 0

    # Convert numpy array back to PIL image
    noisy_image = Image.fromarray(image_array)

    return noisy_image


def add_speckle_noise(image, amount=0.5):
    """
    Add speckle noise to an image
    :param image: PIL Image object
    :param amount: Amount of noise to add
    :return: Noisy image as a PIL Image object
    """

    # Convert image to numpy array (pixels)
    image_array = np.array(image)

    # Add noise
    noise = np.random.randn(*image_array.shape) * amount
    noisy_image_array = image_array + image_array * noise

    # Image boundaries
    noisy_image_array = np.clip(noisy_image_array, 0, 255)

    # Convert numpy array back to PIL image
    noisy_image = Image.fromarray(np.uint8(noisy_image_array))

    return noisy_image
