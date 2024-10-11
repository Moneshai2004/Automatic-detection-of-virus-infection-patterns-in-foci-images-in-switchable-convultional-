import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2
import os

def load_and_preprocess_data(data_dir, img_size=(256, 256), batch_size=32):
    # Creating an ImageDataGenerator with binary class mode
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',  # Set class mode to binary
        subset='training',
        shuffle=True
    )

    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',  # Set class mode to binary
        subset='validation',
        shuffle=True
    )

    return train_generator, validation_generator

def create_switchable_conv_model(input_size=(256, 256, 3)):
    inputs = layers.Input(input_size)

    # Example Switchable Convolution Layer
    x = layers.Conv2D(64, (3, 3), padding='same')(inputs)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Additional layers can be added for complexity
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(256, (3, 3), padding='same')(x)
    x = layers.Activation('relu')(x)

    # Flatten and fully connected layers for classification
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)  # Binary classification

    model = models.Model(inputs, outputs)
    return model

def train_model(model, train_data, validation_data, epochs=10):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Adding ModelCheckpoint to save the best model
    checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', mode='min')

    history = model.fit(
        train_data,
        epochs=epochs,
        validation_data=validation_data,
        callbacks=[checkpoint]  # Add the checkpoint callback
    )
    return history

def detect_infection(model, image_path, img_size=(256, 256)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    prediction = model.predict(img)
    return prediction[0][0]  # Return the probability of infection

def generate_grad_cam(model, img, img_size, layer_name='conv2d_2'):
    # Get model's last convolutional layer
    grad_model = models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )

    # Compute the gradient of the output with respect to the feature map
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)

    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)  # ReLU and normalize
    return heatmap  # No need to call .numpy()

def overlay_heatmap(img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    overlay = cv2.addWeighted(img, alpha, heatmap, 1 - alpha, 0)
    return overlay

def main():
    data_dir = '/home/moni/Desktop/virus-infection-detection/data'
    img_size = (256, 256)

    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' does not exist.")
        return

    train_data, validation_data = load_and_preprocess_data(data_dir, img_size=img_size)

    model = create_switchable_conv_model(input_size=img_size + (3,))
    history = train_model(model, train_data, validation_data)

    # Save the model
    model.save('virus_infection_detection_model.keras')  # Save in Keras format

    # Test the model with a specific image
    test_image_path = '/home/moni/Desktop/virus-infection-detection/data/infected/1.jpg'  # Update this path as needed
    if os.path.exists(test_image_path):
        infection_prob = detect_infection(model, test_image_path, img_size)

        # Output the probability
        print(f"Infection Probability for the test image: {infection_prob:.2f}")

        # Load and preprocess the image for Grad-CAM
        original_img = cv2.imread(test_image_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        original_img_resized = cv2.resize(original_img, img_size)
        img_array = original_img_resized / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Generate Grad-CAM heatmap
        heatmap = generate_grad_cam(model, img_array, img_size)

        # Overlay heatmap on the original image
        cam_img = overlay_heatmap(original_img_resized, heatmap)

        # Display the result
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(original_img_resized)
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(cam_img)
        plt.title(f"Infection Probability: {infection_prob:.2f}")
        plt.axis('off')

        # Save the plot
        plt.savefig('output_plot_with_cam.png')  
        plt.close()  # Close the plot to free up memory
    else:
        print(f"Error: Test image '{test_image_path}' does not exist.")

if __name__ == "__main__":
    main()
