#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

train_df = pd.read_csv('train_v2.csv')
submission_df = pd.read_csv('sample_submission_v2.csv')

print(train_df.head())
print(submission_df.head())


# In[3]:


# Count the occurrences of each label
label_counts = train_df['tags'].str.split(' ').explode().value_counts()

# Print the label counts
print(label_counts)


# In[4]:


import matplotlib.pyplot as plt

# Plot the label distribution
plt.figure(figsize=(12, 6))
label_counts.plot(kind='bar')
plt.title('Label Distribution in Training Data')
plt.xlabel('Label')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()



# In[5]:


from sklearn.model_selection import train_test_split

# Split the data into 80% train and 20% validation/test
train_df, val_test_df = train_test_split(train_df, test_size=0.2, random_state=42)

# Split the validation/test data into 50% validation and 50% test
val_df, test_df = train_test_split(val_test_df, test_size=0.5, random_state=42)

print(train_df.shape)
print(val_df.shape)
print(test_df.shape)


# In[6]:


# Extract image names and construct paths
train_image_names = train_df['image_name'].tolist()
train_image_paths = ['train-jpg/' + name + '.jpg' for name in train_image_names]

val_image_names = val_df['image_name'].tolist()
val_image_paths = ['train-jpg/' + name + '.jpg' for name in val_image_names]

test_image_names = test_df['image_name'].tolist()
test_image_paths = ['train-jpg/' + name + '.jpg' for name in test_image_names]

# Define all possible labels
possible_labels = ['agriculture', 'artisinal_mine', 'bare_ground', 'blooming', 'blow_down', 'clear', 'cloudy', 'conventional_mine', 'cultivation', 'habitation', 'haze', 'partly_cloudy', 'primary', 'road', 'selective_logging', 'slash_burn', 'water']

# Multi-hot encode labels
train_labels = []
for index, row in train_df.iterrows():
    labels = row['tags'].split()
    label_vector = [1 if label in labels else 0 for label in possible_labels]
    train_labels.append(label_vector)

val_labels = []
for index, row in val_df.iterrows():
    labels = row['tags'].split()
    label_vector = [1 if label in labels else 0 for label in possible_labels]
    val_labels.append(label_vector)

test_labels = []
for index, row in test_df.iterrows():
    labels = row['tags'].split()
    label_vector = [1 if label in labels else 0 for label in possible_labels]
    test_labels.append(label_vector)



# In[7]:


# Check for missing values in 'tags' column
print(train_df['tags'].isnull().sum())
print(val_df['tags'].isnull().sum())
print(test_df['tags'].isnull().sum())

# Check for unique labels
all_labels = []
for df in [train_df, val_df, test_df]:
    all_labels.extend(df['tags'].str.split(' ').explode().unique())
unique_labels = set(all_labels)
print(unique_labels)



# In[8]:


def dark_channel_prior(images, patch_size=15, omega=0.95):
    """
    Implements a simple version of the dark channel prior method for haze removal.

    Args:
        images: A batch of images in the format [batch_size, height, width, channels].
        patch_size: The size of the patch to use for finding the dark channel.
        omega: The amount of haze to remove.

    Returns:
        A batch of haze-free images.
    """

    # Find the dark channel
    dark_channel = tf.nn.max_pool(
        -images,
        ksize=[1, patch_size, patch_size, 1],
        strides=[1, 1, 1, 1],
        padding='SAME'
    )

    # Estimate the atmospheric light
    atmospheric_light = tf.reduce_max(tf.reshape(images, [-1, 3]), axis=0)

    # Recover the haze-free image
    transmission = 1 - omega * dark_channel / atmospheric_light
    haze_free_images = (images - atmospheric_light) / transmission + atmospheric_light

    return haze_free_images



# In[9]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create an ImageDataGenerator with desired augmentations
datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.2,
    fill_mode='nearest'
)



# In[10]:


import tensorflow as tf

# Assuming image size is (128, 128, 3) based on common practices
IMG_SIZE = (128, 128)

# Load pre-trained ResNet50 model without the top classification layer
base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

# Freeze the base model layers
for layer in base_model.layers[:-5]:  # Unfreeze the last 5 layers for fine-tuning
    layer.trainable = False

# Add a classification head
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
predictions = tf.keras.layers.Dense(17, activation='sigmoid')(x)  # 17 output classes

# Create the final model
model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

print(model.summary())


# In[ ]:




