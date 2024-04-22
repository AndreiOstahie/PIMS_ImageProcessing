import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageEnhance
import numpy as np
import cv2
import bm3d
import os

class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Image Processing App')

        # Frame for buttons and slider
        frame = tk.Frame(self.root)
        frame.pack(pady=20)

        # Load and Save buttons
        btn_load = tk.Button(frame, text="Load Image", command=self.load_image)
        btn_load.grid(row=0, column=0, padx=10)

        btn_save = tk.Button(frame, text="Save Image", command=self.save_image)
        btn_save.grid(row=0, column=1, padx=10)

        # Reset button
        btn_reset = tk.Button(frame, text="Reset", command=self.reset_image)
        btn_reset.grid(row=0, column=2, padx=10)

        # Autocorrect buttons
        btn_autocorrect = tk.Button(frame, text="Autocorrect", command=self.auto_correct_image)
        btn_autocorrect.grid(row=0, column=3, padx=10)

        btn_autocorrect_incl_noise = tk.Button(frame, text="Autocorrect + Denoise", command=self.auto_correct_image_incl_noise)
        btn_autocorrect_incl_noise.grid(row=0, column=4, padx=10)



        # Noise buttons
        # btn_gaussian = tk.Button(frame, text="Gaussian", command=self.apply_gaussian_noise)
        # btn_gaussian.grid(row=1, column=0, padx=10, pady=10)
        #
        # btn_sp = tk.Button(frame, text="S&P", command=self.apply_salt_and_pepper_noise)
        # btn_sp.grid(row=1, column=1, padx=10)

        # Noise options
        self.noise_type = tk.StringVar(self.root)
        self.noise_type.set("Add Noise")  # default value
        noise_menu = tk.OptionMenu(frame, self.noise_type, "Gaussian", "S&P", "Speckle", command=self.apply_noise)
        noise_menu.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

        # Denoise options
        self.denoise_type = tk.StringVar(self.root)
        self.denoise_type.set("Select Denoise Method")  # default value
        denoise_menu = tk.OptionMenu(frame, self.denoise_type, "Median Filter", "Gaussian Filter", "NLMD", "Bilateral Filter",
                                     command=self.apply_denoise)
        denoise_menu.grid(row=1, column=10, columnspan=2, padx=10, pady=10)

        # Contrast slider
        self.contrast_slider = tk.Scale(frame, from_=0.1, to=2.0, resolution=0.1, orient=tk.HORIZONTAL, label="Adjust Contrast")
        self.contrast_slider.set(1.0)  # Set the default contrast value to 1.0
        self.contrast_slider.grid(row=2, column=0, columnspan=2, pady=10)
        self.contrast_slider.bind("<ButtonRelease-1>", self.adjust_contrast)

        # Brightness slider
        self.brightness_slider = tk.Scale(frame, from_=0.1, to=2.0, resolution=0.1, orient=tk.HORIZONTAL, label="Adjust Brightness")
        self.brightness_slider.set(1.0)  # Set the default brightness value to 1.0
        self.brightness_slider.grid(row=2, column=5, columnspan=2, pady=10)
        self.brightness_slider.bind("<ButtonRelease-1>", self.adjust_brightness)

        # Image display area
        self.img_label = tk.Label(root)
        self.img_label.pack(pady=20)

        # Placeholder for the Image object and the modified version
        self.original_image = None  # Original loaded image
        self.base_image = None  # State of the image after major changes (noise addition)
        self.image = None  # This will now hold the currently displayed (modified) image


    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.original_image = Image.open(path)
            self.base_image = self.original_image.copy()  # Create base image
            self.image = self.original_image.copy()  # Create a copy for modifications
            self.display_image(self.image)

    def save_image(self):
        if self.image:
            path = filedialog.asksaveasfilename(defaultextension=".png",
                                                filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"),
                                                           ("All files", "*.*")])
            if path:
                self.image.save(path)

    def reset_image(self):
        if self.original_image:
            self.contrast_slider.set(1.0)  # Reset contrast slider value to 1.0
            self.brightness_slider.set(1.0)  # Reset brightness slider value to 1.0
            self.base_image = self.original_image
            self.image = self.original_image
            self.display_image(self.image)

    def adjust_contrast(self, event=None):
        if self.base_image:
            enhancer = ImageEnhance.Contrast(self.base_image)
            self.image = enhancer.enhance(self.contrast_slider.get())
            self.display_image(self.image)

    def adjust_brightness(self, event=None):
        if self.base_image:
            enhancer = ImageEnhance.Brightness(self.base_image)
            self.image = enhancer.enhance(self.brightness_slider.get())
            self.display_image(self.image)

    def display_image(self, img):
        img_tk = ImageTk.PhotoImage(img)
        self.img_label.imgtk = img_tk  # Keep a reference!
        self.img_label.config(image=img_tk)
        self.img_label.pack(pady=20)

    def add_gaussian_noise(self, image, mean=0, std=25):
        """
        Add Gaussian noise to an image.
        :param image: PIL Image object
        :param mean: Mean of the Gaussian noise
        :param std: Standard deviation of the Gaussian noise
        :return: Noisy image as a PIL Image object
        """
        # Convert image to array
        image_array = np.array(image)

        # Generate Gaussian noise
        noise = np.random.normal(mean, std, image_array.shape)

        # Add the Gaussian noise to the image
        noisy_image_array = image_array + noise

        # Ensure we respect image boundaries
        noisy_image_array = np.clip(noisy_image_array, 0, 255)

        # Convert back to PIL image
        noisy_image = Image.fromarray(np.uint8(noisy_image_array))
        return noisy_image

    def add_salt_and_pepper_noise(self, image, salt_prob=0.01, pepper_prob=0.01):
        """
        Add salt-and-pepper noise to an image.
        :param image: PIL Image object
        :param salt_prob: Probability of adding salt noise
        :param pepper_prob: Probability of adding pepper noise
        :return: Noisy image as a PIL Image object
        """
        # Convert image to array
        image_array = np.array(image)

        # Salt noise
        num_salt = np.ceil(salt_prob * image_array.size * 0.5)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image_array.shape]
        image_array[tuple(coords)] = 255

        # Pepper noise
        num_pepper = np.ceil(pepper_prob * image_array.size * 0.5)
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image_array.shape]
        image_array[tuple(coords)] = 0

        # Convert back to PIL image
        noisy_image = Image.fromarray(image_array)
        return noisy_image

    def add_speckle_noise(self, image, amount=0.5):
        """
        Add speckle noise to an image.
        :param image: PIL Image object
        :param amount: Amount of noise to add
        :return: Noisy image as a PIL Image object
        """
        image_array = np.array(image)
        noise = np.random.randn(*image_array.shape) * amount
        noisy_image_array = image_array + image_array * noise
        noisy_image_array = np.clip(noisy_image_array, 0, 255)
        noisy_image = Image.fromarray(np.uint8(noisy_image_array))
        return noisy_image




    def apply_gaussian_noise(self):
        if self.image:
            self.image = self.add_gaussian_noise(self.image)
            self.base_image = self.image.copy()  # Update the baseline image after applying noise
            self.display_image(self.image)

    def apply_salt_and_pepper_noise(self):
        if self.image:
            self.image = self.add_salt_and_pepper_noise(self.image)
            self.base_image = self.image.copy()  # Update the baseline image after applying noise
            self.display_image(self.image)

    def apply_noise(self, choice):
        if self.image:
            if choice == "Gaussian":
                self.image = self.add_gaussian_noise(self.image)
            elif choice == "S&P":
                self.image = self.add_salt_and_pepper_noise(self.image)
            elif choice == "Speckle":
                self.image = self.add_speckle_noise(self.image)

            self.contrast_slider.set(1.0)  # Reset contrast slider value to 1.0
            self.brightness_slider.set(1.0)  # Reset brightness slider value to 1.0
            self.base_image = self.image.copy()  # Update the baseline image after applying noise
            self.display_image(self.image)

    def apply_denoise(self, method):
        if self.image:
            if method == "Median Filter":
                self.image = self.median_filter(self.image)
            elif method == "Gaussian Filter":
                self.image = self.gaussian_filter(self.image)
            elif method == "NLMD":
                self.image = self.non_local_means_denoising(self.image)
            elif method == "Bilateral Filter":
                self.image = self.bilateral_filter_denoising(self.image)

            self.contrast_slider.set(1.0)  # Reset contrast slider value to 1.0
            self.brightness_slider.set(1.0)  # Reset brightness slider value to 1.0
            self.base_image = self.image.copy()  # Update the baseline image after applying denoise
            self.display_image(self.image)



    # def median_filter(self, image):
    #     """ Apply median filter to denoise the image """
    #     image_array = np.array(image)
    #     denoised_array = cv2.medianBlur(image_array, 5)  # using a 5x5 kernel
    #     denoised_image = Image.fromarray(denoised_array)
    #     return denoised_image
    #
    # def gaussian_filter(self, image):
    #     """ Apply Gaussian filter to denoise the image """
    #     image_array = np.array(image)
    #     denoised_array = cv2.GaussianBlur(image_array, (5, 5), 0)  # using a 5x5 kernel
    #     denoised_image = Image.fromarray(denoised_array)
    #     return denoised_image
    #
    # def non_local_means_denoising(self, image):
    #     """ Apply Non-local Means denoising algorithm to an image """
    #     image_array = np.array(image)
    #     denoised_array = cv2.fastNlMeansDenoisingColored(image_array, None, 10, 10, 7, 21)
    #     denoised_image = Image.fromarray(denoised_array)
    #     return denoised_image



    def median_filter(self, image):
        """
        Apply median filter to denoise the image using OpenCV, properly handling RGB to BGR conversion.

        Parameters:
        image: PIL Image object in RGB format.

        Returns:
        denoised_image: A denoised PIL Image object in RGB format.
        """
        # Convert the PIL Image to a NumPy array (RGB)
        image_array = np.array(image)

        # Convert RGB to BGR for OpenCV processing
        image_array_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        # Apply the median filter
        denoised_array_bgr = cv2.medianBlur(image_array_bgr, 5)  # using a 5x5 kernel

        # Convert the denoised image back from BGR to RGB
        denoised_array_rgb = cv2.cvtColor(denoised_array_bgr, cv2.COLOR_BGR2RGB)

        # Convert the denoised NumPy array back to a PIL Image
        denoised_image = Image.fromarray(denoised_array_rgb)

        return denoised_image

    def gaussian_filter(self, image):
        """
        Apply Gaussian filter to denoise the image using OpenCV, properly handling RGB to BGR conversion.

        Parameters:
        image: PIL Image object in RGB format.

        Returns:
        denoised_image: A denoised PIL Image object in RGB format.
        """
        # Convert the PIL Image to a NumPy array (RGB)
        image_array = np.array(image)

        # Convert RGB to BGR for OpenCV processing
        image_array_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        # Apply the Gaussian filter
        denoised_array_bgr = cv2.GaussianBlur(image_array_bgr, (5, 5), 0)  # using a 5x5 kernel

        # Convert the denoised image back from BGR to RGB
        denoised_array_rgb = cv2.cvtColor(denoised_array_bgr, cv2.COLOR_BGR2RGB)

        # Convert the denoised NumPy array back to a PIL Image
        denoised_image = Image.fromarray(denoised_array_rgb)

        return denoised_image

    def non_local_means_denoising(self, image):
        """
            Apply Non-local Means denoising algorithm to an image using OpenCV.
            This function assumes the input image is a PIL Image in RGB format.

            Parameters:
            image: PIL Image object in RGB format.

            Returns:
            denoised_image: A denoised PIL Image object in RGB format.
            """
        # Convert the PIL Image to a NumPy array (RGB)
        image_array = np.array(image)

        # Convert RGB to BGR for OpenCV processing
        image_array_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        # Apply the Non-local Means denoising algorithm
        denoised_array_bgr = cv2.fastNlMeansDenoisingColored(image_array_bgr, None, 10, 10, 7, 21)

        # Convert the denoised image back from BGR to RGB
        denoised_array_rgb = cv2.cvtColor(denoised_array_bgr, cv2.COLOR_BGR2RGB)

        # Convert the denoised NumPy array back to a PIL Image
        denoised_image = Image.fromarray(denoised_array_rgb)

        return denoised_image

    def bilateral_filter_denoising(self, image, diameter=9, sigma_color=75, sigma_space=75):
        """
        Apply bilateral filtering to a PIL Image object for noise reduction while preserving edges.

        Parameters:
        - image: PIL Image object.
        - diameter: Diameter of each pixel neighborhood that is used during filtering.
        - sigma_color: Filter sigma in the color space.
        - sigma_space: Filter sigma in the coordinate space.

        Returns:
        - denoised_image: The denoised image as a PIL Image object.
        """
        # Convert PIL Image to a NumPy array
        image_array = np.array(image)

        # Check if the image is grayscale or color
        if len(image_array.shape) == 2:  # Grayscale image, no channel dimension
            denoised_image_array = cv2.bilateralFilter(image_array, diameter, sigma_color, sigma_space)
        else:  # Color image, assume it's RGB
            # Convert RGB to BGR for OpenCV processing
            image_array_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            denoised_image_array_bgr = cv2.bilateralFilter(image_array_bgr, diameter, sigma_color, sigma_space)
            # Convert back from BGR to RGB
            denoised_image_array = cv2.cvtColor(denoised_image_array_bgr, cv2.COLOR_BGR2RGB)

        # Convert back to PIL Image
        denoised_image = Image.fromarray(denoised_image_array)

        return denoised_image

    def auto_correct_image_incl_noise(self):
        # Convert image
        image = None
        if self.image:
            image = self.image
        image_array = np.array(image)

        # Step 1: Analyze the image
        mean_intensity = np.mean(image_array)
        std_intensity = np.std(image_array)

        # Noise level estimation (simplified using variance)
        noise_estimate = np.var(image_array)

        # Step 2: Apply corrections

        # Denoising if noise is above a threshold
        if noise_estimate > 500:  # Threshold is arbitrary; adjust based on your needs
            # Assuming bm3d.bm3d returns a numpy array, ensure proper function or library import for bm3d
            denoised_array = bm3d.bm3d(image_array, sigma_psd=noise_estimate / 255,
                                       stage_arg=bm3d.BM3DStages.ALL_STAGES)
            # Convert denoised array back to PIL Image for further processing
            image = Image.fromarray(denoised_array.astype(np.uint8))

        # Brightness adjustment
        if mean_intensity < 128:  # Assuming the lower half is too dark
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(2 - mean_intensity / 128)

        # Contrast adjustment
        if std_intensity < 50:  # Arbitrary threshold for low contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2)

        self.display_image(image)

        return image




    def auto_correct_image(self):
        # Convert image
        image = None
        if self.image:
            image = self.image
        image_array = np.array(image)

        # Brightness adjustment
        # Calculate the mean intensity to determine if the image is too dark or too bright.
        mean_intensity = np.mean(image_array)
        if mean_intensity < 128:  # If the image is too dark
            enhancer = ImageEnhance.Brightness(image)
            # Increase brightness. The factor depends inversely on how dark the image is.
            image = enhancer.enhance(2 - mean_intensity / 128)
        elif mean_intensity > 192:  # If the image is too bright (assuming 192 as a threshold for "too bright")
            enhancer = ImageEnhance.Brightness(image)
            # Decrease brightness. The factor is calculated to lower brightness based on how bright the image is.
            image = enhancer.enhance(1 - (mean_intensity - 192) / 64)

        # Contrast adjustment
        # Calculate the standard deviation to check the contrast level.
        std_intensity = np.std(image_array)
        if std_intensity < 50:  # If contrast is too low, irrespective of brightness
            enhancer = ImageEnhance.Contrast(image)
            # Increase contrast more significantly if it's very low.
            image = enhancer.enhance(2)
        elif std_intensity > 70:  # If contrast is too high (assuming 70 as a high threshold)
            enhancer = ImageEnhance.Contrast(image)
            # Decrease contrast subtly.
            image = enhancer.enhance(0.8)

        self.display_image(image)

        return image

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()
