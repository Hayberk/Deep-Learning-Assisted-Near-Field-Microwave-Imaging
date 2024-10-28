import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib.colors import Normalize
import time


def load_data(data_dir):
    noisy_images = []
    clean_images = []

    noisy_dir = os.path.join(data_dir, 'SVD_noisy')
    clean_dir = os.path.join(data_dir, 'ground-truth')

    # Load and process noisy images
    for filename in os.listdir(noisy_dir):
        img_path = os.path.join(noisy_dir, filename)
        img = Image.open(img_path).convert('L')
        img_array = np.array(img) / 255.0
        noisy_images.append(img_array.reshape((64, 64, 1)))

    # Load and process clean (ground truth) images
    for filename in os.listdir(clean_dir):
        img_path = os.path.join(clean_dir, filename)
        img = Image.open(img_path).convert('L')
        img_array = np.array(img) / 255.0
        clean_images.append(img_array.reshape((64, 64, 1)))

    print("Pictures are loaded.")
    return np.array(noisy_images), np.array(clean_images)


# Define U-net model with increased connections
def unet_model(input_shape=(64, 64, 1)):
    inputs = Input(input_shape)

    # Encoder
    conv1 = Conv2D(128, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(128, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(256, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(256, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(512, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(512, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Central part
    conv4 = Conv2D(1024, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(1024, 3, activation='relu', padding='same')(conv4)

    # Decoder
    up5 = Concatenate()([UpSampling2D(size=(2, 2))(conv4), conv3])
    conv5 = Conv2D(512, 3, activation='relu', padding='same')(up5)
    conv5 = Conv2D(512, 3, activation='relu', padding='same')(conv5)

    up6 = Concatenate()([UpSampling2D(size=(2, 2))(conv5), conv2])
    conv6 = Conv2D(256, 3, activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, 3, activation='relu', padding='same')(conv6)

    up7 = Concatenate()([UpSampling2D(size=(2, 2))(conv6), conv1])
    conv7 = Conv2D(128, 3, activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, 3, activation='relu', padding='same')(conv7)

    # Output layer
    outputs = Conv2D(1, 1, activation='sigmoid')(conv7)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model


# Function to map pixel values to dielectric coefficients
def pixel_to_dielectric(pixel_value):
    return 2 + (18 / 255) * pixel_value


# Directory where your dataset is located
data_dir = r"C:\Users\MR Robot\Desktop\Grad_Project"

# Load the data
noisy_images, clean_images = load_data(data_dir)

# Split dataset into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(noisy_images, clean_images, test_size=0.3, random_state=42)

print("Number of training images:", x_train.shape[0])
print("Number of validation images:", x_val.shape[0])

# Create and compile the model
model = unet_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Set up EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Start training
print("Training has started.")
start_time = time.time()

# Train the model with EarlyStopping
history = model.fit(x_train, y_train, epochs=100, batch_size=5, validation_data=(x_val, y_val),
                    callbacks=[early_stopping])

# Save the model
model_save_path = "unet_model_v_1.h5"
save_model(model, model_save_path)
print(f"Model saved to {model_save_path}")
print("The trained model is saved.")

# End training
end_time = time.time()
print("Training has completed.")
print(f"Total time: {end_time - start_time:.2f} seconds")

# Plot training & validation loss values
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='green')
plt.title('Training & Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot training & validation accuracy values
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy', color='red')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
plt.title('Training & Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Predict on validation set
y_pred = model.predict(x_val)

# Save grayscale and colored output images
gray_output_dir = "Unet_Output_with_gray_scale"
color_output_dir = "Unet_Output_with_colour"
os.makedirs(gray_output_dir, exist_ok=True)
os.makedirs(color_output_dir, exist_ok=True)

# Save all validation images in grayscale and color
for i in range(len(x_val)):
    input_image = x_val[i].squeeze()
    ground_truth = y_val[i].squeeze()
    output_image = y_pred[i].squeeze()

    # Convert output image to dielectric coefficient
    dielectric_output = pixel_to_dielectric(output_image * 255)

    # Save grayscale output image
    gray_output_path = os.path.join(gray_output_dir, f"output_image_{i}.png")
    plt.imsave(gray_output_path, output_image, cmap='gray')

    # Normalize the dielectric data for color plotting
    norm = Normalize(vmin=2, vmax=20)

    # Plot and save colored output image, fixing upside-down issue
    plt.figure(figsize=(6, 6))
    plt.pcolormesh(np.flipud(dielectric_output), cmap='jet', norm=norm)
    plt.colorbar(label='Dielectric Coefficient')
    plt.title('Dielectric Coefficient Map')
    plt.axis('equal')

    color_output_path = os.path.join(color_output_dir, f"color_output_image_{i}.png")
    plt.savefig(color_output_path, format='png')
    plt.close()  # Close the figure after saving

    # Display progress
    if (i + 1) % 10 == 0:
        print(f"Processed {i + 1} images out of {len(x_val)}")

print("All images saved successfully.")

# Calculate SSIM for each pair of input and ground truth images
ssim_input_gt = [ssim(x_val[i].squeeze(), y_val[i].squeeze(), data_range=1) for i in range(len(x_val))]

# Calculate SSIM for each pair of output and ground truth images
ssim_output_gt = [ssim(y_pred[i].squeeze(), y_val[i].squeeze(), data_range=1) for i in range(len(y_pred))]

# Compute average SSIM for input and ground truth
average_ssim_input_gt = np.mean(ssim_input_gt)
print("Average SSIM (Input vs. Ground Truth):", average_ssim_input_gt)

# Compute average SSIM for output and ground truth
average_ssim_output_gt = np.mean(ssim_output_gt)
print("Average SSIM (Output vs. Ground Truth):", average_ssim_output_gt)

# Compute SSIM differences for average calculation
ssim_diff = np.array(ssim_output_gt) - np.array(ssim_input_gt)
average_ssim_diff = np.mean(ssim_diff)
print("Average SSIM Difference (Output vs. Input):", average_ssim_diff)

# Display some sample images along with SSIM values
num_samples = 10
selected_indices = np.random.choice(len(x_val), num_samples, replace=False)

for i in selected_indices:
    input_image = x_val[i].squeeze()
    ground_truth = y_val[i].squeeze()
    output_image = y_pred[i].squeeze()

    ssim_input_gt_value = ssim_input_gt[i]
    ssim_output_gt_value = ssim_output_gt[i]

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 4, 1)
    plt.imshow(input_image, cmap='gray')
    plt.title('Input Image')
    plt.axis('off')

    plt.subplot(2, 4, 2)
    plt.imshow(output_image, cmap='gray')
    plt.title('Output Image (Gray)')
    plt.axis('off')

    plt.subplot(2, 4, 3)
    plt.imshow(ground_truth, cmap='gray')
    plt.title('Ground Truth')
    plt.axis('off')

    plt.subplot(2, 4, 4)
    dielectric_output = pixel_to_dielectric(output_image * 255)
    norm = Normalize(vmin=2, vmax=20)
    plt.pcolormesh(np.flipud(dielectric_output), cmap='jet', norm=norm)
    plt.colorbar(label='Dielectric Coefficient')
    plt.title(f'Output Image (Color)\n(Brightness Range: {norm.vmin} - {norm.vmax})')
    plt.axis('equal')

    plt.subplot(2, 4, 5)
    plt.text(0.5, 0.5, f'Input vs. GT SSIM: {ssim_input_gt_value:.4f}', fontsize=10, ha='center')
    plt.axis('off')

    plt.subplot(2, 4, 6)
    plt.text(0.5, 0.5, f'Output vs. GT SSIM: {ssim_output_gt_value:.4f}', fontsize=10, ha='center')
    plt.axis('off')

    plt.subplot(2, 4, 7)
    plt.text(0.5, 0.5, f'SSIM Difference: {ssim_diff[i]:.4f}', fontsize=10, ha='center')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
