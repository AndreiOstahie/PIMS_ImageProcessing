import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageEnhance
import numpy as np
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

        # Contrast slider
        self.slider = tk.Scale(frame, from_=0.1, to=2.0, resolution=0.1, orient=tk.HORIZONTAL, label="Adjust Contrast")
        self.slider.set(1.0)  # Set the default contrast value to 1.0
        self.slider.grid(row=2, column=0, columnspan=2, pady=10)
        self.slider.bind("<ButtonRelease-1>", self.adjust_contrast)

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
            self.slider.set(1.0)  # Reset contrast slider value to 1.0
            self.base_image = self.original_image
            self.image = self.original_image
            self.display_image(self.image)

    def adjust_contrast(self, event=None):
        if self.base_image:
            enhancer = ImageEnhance.Contrast(self.base_image)
            self.image = enhancer.enhance(self.slider.get())
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

            self.slider.set(1.0)  # Reset contrast slider value to 1.0
            self.base_image = self.image.copy()  # Update the baseline image after applying noise
            self.display_image(self.image)



if __name__ == "__main__":
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()
