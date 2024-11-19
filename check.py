import os
import shutil
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2


# Paths to the dataset
source_dir = "/Users/sahilv/Downloads/archive (1)/food-101/food-101/images"
dest_dir = "food101_subset"
num_images = 100

# Create the destination directory
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

# Get the classes (food categories)
classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

# Determine the number of images to select per class
images_per_class = num_images // len(classes)
remaining_images = num_images % len(classes)

# Copy selected images to the new subset directory
for cls in classes:
    cls_source_dir = os.path.join(source_dir, cls)
    cls_dest_dir = os.path.join(dest_dir, cls)
    
    if not os.path.exists(cls_dest_dir):
        os.makedirs(cls_dest_dir)
    
    files = os.listdir(cls_source_dir)
    images = [img for img in files if img.endswith(".jpg")]
    
    num_to_select = images_per_class + (1 if remaining_images > 0 else 0)
    
    selected_images = random.sample(images, min(len(images), num_to_select))
    
    remaining_images -= len(selected_images) - images_per_class
    
    for img in selected_images:
        src_path = os.path.join(cls_source_dir, img)
        dest_path = os.path.join(cls_dest_dir, img)
        shutil.copy(src_path, dest_path)

# Load the pre-trained MobileNetV2 model without the top layers (for fine-tuning)
base_model = MobileNetV2(weights='imagenet', input_shape=(150, 150, 3), include_top=False)

# Freeze the base model layers initially
base_model.trainable = False
num_classes = len(classes)

# Build the model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)),  # L2 regularization
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')  # Adjusting the output layer to match the number of classes
])

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# Data augmentation and image preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2],  # Adjust brightness
    channel_shift_range=20       # Randomly shift color channels
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Prepare the image generators
train_generator = train_datagen.flow_from_directory(
    dest_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    dest_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# Learning rate scheduler
def lr_schedule(epoch):
    return 0.001

lr_scheduler = LearningRateScheduler(lr_schedule)

# Early stopping to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# TensorBoard for better monitoring
tensorboard = TensorBoard(log_dir='./logs')

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=30,  # Increase epochs to allow the model to converge
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=[lr_scheduler, early_stopping, tensorboard]
)

# After training, unfreeze the last few layers of the model for fine-tuning
base_model.trainable = True

# Unfreeze all layers except the last 10 layers
for layer in base_model.layers[:-10]:
    layer.trainable = False

# Recompile the model after unfreezing layers
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# Continue training the model with the newly unfrozen layers
history_finetune = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=50,  # Continue for more epochs
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=[lr_scheduler, early_stopping, tensorboard]
)

# Evaluate the model on validation data
val_loss, val_acc = model.evaluate(validation_generator, steps=validation_generator.samples // validation_generator.batch_size)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_acc}")

def predict_image(image_path, answer):
    # Load the image, resizing it to the target size (150, 150)
    img = load_img(image_path, target_size=(150, 150))
    
    # Convert the image to a numpy array and scale the pixel values to [0, 1]
    img_array = img_to_array(img) / 255.0
    
    # Add an extra dimension for the batch size (since the model expects a batch of images)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make the prediction
    predictions = model.predict(img_array)
    
    # Get the predicted class
    predicted_class = np.argmax(predictions, axis=1)
    class_name = classes[predicted_class[0]]  # Assuming 'classes' contains the names of your categories
    
    print(f"The predicted class is: {class_name} but it's supposed to be: {answer}")

# Example usage: Predict on a sample image
image_path = "/Users/sahilv/Downloads/baklava.jpg" 
image_path1 = '/Users/sahilv/Downloads/redvelvet.jpg'
image_path2 = '/Users/sahilv/Downloads/breakfastburrito.webp'
predict_image(image_path, "baklava")
predict_image(image_path1, "red velvet cake")
predict_image(image_path2, "breakfast burrito")
