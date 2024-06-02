import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageEnhance
import numpy as np
import cv2
import bm3d
import os
from noise import add_gaussian_noise, add_salt_and_pepper_noise, add_speckle_noise
from denoise import median_filter, gaussian_filter, non_local_means, bilateral_filter
from denoise import median_filter_2, gaussian_filter_2, non_local_means_2, bilateral_filter_2


# Noise and denoise strength
GAUSSIAN_NOISE_STRENGTHS = [5, 10, 15, 20, 25]
SP_NOISE_STRENGTHS = [0.01, 0.05, 0.075, 0.1, 0.125]
SPECKLE_NOISE_STRENGTHS = [0.1, 0.2, 0.3, 0.4, 0.5]

MEDIAN_DENOISE_STRENGTHS = [3, 5, 7, 11, 15]
GAUSSIAN_DENOISE_STRENGTHS = [5, 7, 11, 15, 19]
NLMD_DENOISE_STRENGTHS = [5, 10, 15, 25, 35]
BILATERAL_DENOISE_STRENGTHS = [(7, 50), (9, 75), (11, 100), (15, 125), (19, 150)]


class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Image Processing App')

        # Set initial noise and denoise strength
        self.noise_strength = 1
        self.denoise_strength = 1

        # Frame for window content (image, buttons, sliders etc.)
        frame = tk.Frame(self.root)
        frame.pack(pady=20)

        # Load button
        btn_load = tk.Button(frame, text="Load Image", command=self.load_image)
        btn_load.grid(row=0, column=0, padx=10)

        # Save button
        btn_save = tk.Button(frame, text="Save Image", command=self.save_image)
        btn_save.grid(row=0, column=1, padx=10)

        # Reset button
        btn_reset = tk.Button(frame, text="Reset", command=self.reset_image)
        btn_reset.grid(row=0, column=2, padx=10)

        # Autocorrect buttons
        btn_autocorrect = tk.Button(frame, text="Autocorrect", command=self.auto_correct_image)
        btn_autocorrect.grid(row=0, column=3, padx=10)

        btn_autocorrect_incl_noise = tk.Button(frame, text="Autocorrect + Denoise", command=self.auto_correct_image_including_noise)
        btn_autocorrect_incl_noise.grid(row=0, column=4, padx=10)

        # Contrast slider
        self.contrast_slider = tk.Scale(frame, from_=0.1, to=2.0, resolution=0.1, orient=tk.HORIZONTAL, label="Adjust Contrast")
        self.contrast_slider.set(1.0)  # Set the default contrast value to 1.0
        self.contrast_slider.grid(row=0, column=5, columnspan=2, pady=10)
        self.contrast_slider.bind("<ButtonRelease-1>", self.adjust_contrast)

        # Brightness slider
        self.brightness_slider = tk.Scale(frame, from_=0.1, to=2.0, resolution=0.1, orient=tk.HORIZONTAL, label="Adjust Brightness")
        self.brightness_slider.set(1.0)  # Set the default brightness value to 1.0
        self.brightness_slider.grid(row=0, column=10, columnspan=2, pady=10)
        self.brightness_slider.bind("<ButtonRelease-1>", self.adjust_brightness)

        # Noise options
        self.noise_type = tk.StringVar(self.root)
        self.noise_type.set("Add Noise")  # default value
        noise_menu = tk.OptionMenu(frame, self.noise_type, "Gaussian", "S&P", "Speckle", command=self.apply_noise)
        noise_menu.grid(row=1, column=0, columnspan=1, padx=10, pady=10)

        # Slider for noise strength
        self.noise_slider = tk.Scale(frame, from_=1, to=5, resolution=1, orient=tk.HORIZONTAL, label="Noise Strength")
        self.noise_slider.set(1)  # Set the default noise strength to 1
        self.noise_slider.grid(row=1, column=1, columnspan=1, pady=10)
        self.noise_slider.bind("<ButtonRelease-1>", self.adjust_noise_strength)

        # Denoise options
        self.denoise_type = tk.StringVar(self.root)
        self.denoise_type.set("Denoise Method")  # default value
        denoise_menu = tk.OptionMenu(frame, self.denoise_type, "Median Filter", "Gaussian Filter", "NLMD", "Bilateral Filter",
                                     command=self.apply_denoise)
        denoise_menu.grid(row=1, column=3, columnspan=1, padx=10, pady=10)

        # Slider for denoising strength
        self.denoise_slider = tk.Scale(frame, from_=1, to=5, resolution=1, orient=tk.HORIZONTAL, label="Denoise Strength")
        self.denoise_slider.set(1)  # Set the default denoise strength to 1
        self.denoise_slider.grid(row=1, column=4, columnspan=1, pady=10)
        self.denoise_slider.bind("<ButtonRelease-1>", self.adjust_denoise_strength)

        # Checkbox to toggle between pre-built and custom denoising methods
        # False - use prebuilt functions from libraries (e.g. OpenCV)
        # True - use custom functions
        self.use_custom_denoise = tk.BooleanVar()
        self.use_custom_denoise.set(False)  # Default value: unchecked, False
        check_custom_denoise = tk.Checkbutton(frame, text="Use Custom Denoise", variable=self.use_custom_denoise,
                                              onvalue=True, offvalue=False)
        check_custom_denoise.grid(row=1, column=5, columnspan=1, padx=10)

        # Image display area
        self.img_label = tk.Label(root)
        self.img_label.pack(pady=20)

        # Placeholder for the Image object and the modified version
        self.original_image = None  # Original loaded image
        self.base_image = None  # State of the image after major changes (e.g. adding noise)
        self.image = None  # Currently displayed (modified) image

    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.original_image = Image.open(path)

            # Check if image is grayscale
            is_grayscale = messagebox.askyesno("Grayscale Image", "Is the loaded image grayscale?")
            if is_grayscale:
                self.original_image = self.original_image.convert('L')
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

    def adjust_noise_strength(self, event=None):
        self.noise_strength = self.noise_slider.get()

    def adjust_denoise_strength(self, event=None):
        self.denoise_strength = self.denoise_slider.get()

    def display_image(self, img):
        img_tk = ImageTk.PhotoImage(img)
        self.img_label.imgtk = img_tk
        self.img_label.config(image=img_tk)
        self.img_label.pack(pady=20)

    # Apply different types of noise based on selection
    def apply_noise(self, choice):
        if self.image:
            if choice == "Gaussian":
                strength = GAUSSIAN_NOISE_STRENGTHS[self.noise_strength - 1]
                self.image = add_gaussian_noise(self.image, 0, strength)
            elif choice == "S&P":
                strength = SP_NOISE_STRENGTHS[self.noise_strength - 1]
                self.image = add_salt_and_pepper_noise(self.image, strength, strength)
            elif choice == "Speckle":
                strength = SPECKLE_NOISE_STRENGTHS[self.noise_strength - 1]
                self.image = add_speckle_noise(self.image, strength)

            self.contrast_slider.set(1.0)  # Reset contrast slider value to 1.0
            self.brightness_slider.set(1.0)  # Reset brightness slider value to 1.0
            self.base_image = self.image.copy()  # Update the baseline image after applying noise
            self.display_image(self.image)

    # Apply different types of denoising based on selection
    def apply_denoise(self, method):
        if self.image:
            if method == "Median Filter":
                strength = MEDIAN_DENOISE_STRENGTHS[self.denoise_strength - 1]
                if self.use_custom_denoise.get():
                    self.image = median_filter_2(self.image, strength)
                else:
                    self.image = median_filter(self.image, strength)
            elif method == "Gaussian Filter":
                strength = GAUSSIAN_DENOISE_STRENGTHS[self.denoise_strength - 1]
                if self.use_custom_denoise.get():
                    self.image = gaussian_filter_2(self.image, strength)
                else:
                    self.image = gaussian_filter(self.image, strength)
            elif method == "NLMD":
                strength = NLMD_DENOISE_STRENGTHS[self.denoise_strength - 1]
                if self.use_custom_denoise.get():
                    self.image = non_local_means_2(self.image, strength)
                else:
                    self.image = non_local_means(self.image, strength)
            elif method == "Bilateral Filter":
                diameter = BILATERAL_DENOISE_STRENGTHS[self.denoise_strength - 1][0]
                sigma = BILATERAL_DENOISE_STRENGTHS[self.denoise_strength - 1][1]
                if self.use_custom_denoise.get():
                    self.image = bilateral_filter_2(self.image, diameter, sigma, sigma)
                else:
                    self.image = bilateral_filter(self.image, diameter, sigma, sigma)

            self.contrast_slider.set(1.0)  # Reset contrast slider value to 1.0
            self.brightness_slider.set(1.0)  # Reset brightness slider value to 1.0
            self.base_image = self.image.copy()  # Update the baseline image after applying denoise
            self.display_image(self.image)

    def auto_correct_image(self):
        # Convert image
        image = None
        if self.image:
            image = self.image
        image_array = np.array(image)

        # Brightness adjustment
        # Calculate mean intensity (used to determine if the image is too dark or too bright)
        mean_intensity = np.mean(image_array)
        if mean_intensity < 128:  # Image is too dark
            enhancer = ImageEnhance.Brightness(image)
            # Increase brightness
            image = enhancer.enhance(2 - mean_intensity / 128)
        elif mean_intensity > 192:  # Image is too bright
            enhancer = ImageEnhance.Brightness(image)
            # Decrease brightness
            image = enhancer.enhance(1 - (mean_intensity - 192) / 64)

        # Contrast adjustment
        # Calculate standard deviation (used to check the contrast level)
        std_intensity = np.std(image_array)
        if std_intensity < 50:  # Low contrast
            enhancer = ImageEnhance.Contrast(image)
            # Increase contrast
            image = enhancer.enhance(2)
        elif std_intensity > 70:  # High contrast
            enhancer = ImageEnhance.Contrast(image)
            # Decrease contrast
            image = enhancer.enhance(0.8)

        # Update displayed image
        self.display_image(image)

        return image

    def auto_correct_image_including_noise(self):
        # Convert image
        image = None
        if self.image:
            image = self.image
        image_array = np.array(image)

        # Noise level estimation (using variance)
        noise_estimate = np.var(image_array)

        # Denoise
        if noise_estimate > 500:  # Noise above threshold
            denoised_array = bm3d.bm3d(image_array, sigma_psd=noise_estimate / 255,
                                       stage_arg=bm3d.BM3DStages.ALL_STAGES)
            # Convert denoised array back to PIL Image
            image = Image.fromarray(denoised_array.astype(np.uint8))

        # Brightness adjustment
        # Calculate mean intensity (used to determine if the image is too dark or too bright)
        mean_intensity = np.mean(image_array)
        if mean_intensity < 128:  # Image is too dark
            enhancer = ImageEnhance.Brightness(image)
            # Increase brightness
            image = enhancer.enhance(2 - mean_intensity / 128)
        elif mean_intensity > 192:  # Image is too bright
            enhancer = ImageEnhance.Brightness(image)
            # Decrease brightness
            image = enhancer.enhance(1 - (mean_intensity - 192) / 64)

        # Contrast adjustment
        # Calculate standard deviation (used to check the contrast level)
        std_intensity = np.std(image_array)
        if std_intensity < 50:  # Low contrast
            enhancer = ImageEnhance.Contrast(image)
            # Increase contrast
            image = enhancer.enhance(2)
        elif std_intensity > 70:  # High contrast
            enhancer = ImageEnhance.Contrast(image)
            # Decrease contrast
            image = enhancer.enhance(0.8)

        # Update displayed image
        self.display_image(image)

        return image


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()
