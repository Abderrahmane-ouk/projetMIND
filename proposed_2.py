import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import pandas as pd
import numpy as np
import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPool2D, Dense, Dropout, BatchNormalization, Flatten, GlobalAveragePooling2D, Multiply, Reshape
from keras.layers import concatenate, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Nadam
from sklearn.preprocessing import LabelBinarizer

def convert(y_set):
    letter2pos = {}
    df_letters = pd.read_csv(r'letter_to_pose_2.csv')
    alphabet = 'ABCDEFGHIKLMNOPQRSTUVWXY'
    for i in range(2):
        letter2pos[i] = df_letters[df_letters["Letter"] == alphabet[i]]
    df = pd.concat([letter2pos[y] for y in y_set.iloc], ignore_index=True)
    del df['Letter']

    subst = {'open' : 0., 'semiclosed' : 0.5, 'closed' : 1.,
             'flat' : 0., 'curved' : 0.5, 'bent' : 1.,
             'FAUX' : 0., 'VRAI' : 1.}

    df = df.replace(subst)

    return df

# Load data
train_df = pd.read_csv(r'sign_mnist_train.csv')
test_df = pd.read_csv(r'sign_mnist_test.csv')
# blabla
merged_df = pd.concat([train_df, test_df], ignore_index=True)
merged_df = merged_df[merged_df['label'].isin([0, 1])]
train_df, test_df = train_test_split(merged_df, test_size=0.2, random_state=42)
# On sépare le train en deux parties (train et validation)
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42) 
y = test_df['label']
y_train = convert(train_df['label'])
y_test = convert(test_df['label'])
y_val = convert(val_df['label'])
del train_df['label']
del test_df['label']
del val_df['label']

# Label binarization
label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(y_train)
y_test = label_binarizer.fit_transform(y_test)
y_val = label_binarizer.fit_transform(y_val)
# Normalize and reshape data
x_train = train_df.values / 255
x_test = test_df.values / 255
x_val = val_df.values / 255
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
x_val = x_val.reshape(-1, 28, 28, 1)

# Visualize some samples
f, ax = plt.subplots(2, 5)
f.set_size_inches(10, 10)
k = 0
for i in range(2):
    for j in range(5):
        ax[i, j].imshow(x_train[k].reshape(28, 28), cmap="gray")
        k += 1
    plt.tight_layout()

# Data augmentation
datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False,
    vertical_flip=False
)
datagen.fit(x_train)

# Learning rate reduction callback
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.5, min_lr=0.00001)

# Define the Attention Mechanism
def attention_block(input_tensor):
    attention = Conv2D(1, (1, 1), activation='sigmoid')(input_tensor)
    return Multiply()([input_tensor, attention])

# Define the Nevestro Densenet Attention (SNDA) model
def build_snda(input_shape=(28, 28, 1)):
    inputs = Input(shape=input_shape)
    
    # Initial Conv Layer
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPool2D((2, 2), strides=2, padding='same')(x)
    
    # Dense Block 1
    x1 = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x1 = BatchNormalization()(x1)
    x1 = attention_block(x1)  # Add attention
    x = concatenate([x, x1])
    
    # Dense Block 2
    x2 = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x2 = BatchNormalization()(x2)
    x2 = attention_block(x2)  # Add attention
    x = concatenate([x, x2])
    
    # Transition Layer
    x = Conv2D(256, (1, 1), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D((2, 2), strides=2, padding='same')(x)
    
    # Dense Block 3
    x3 = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x3 = BatchNormalization()(x3)
    x3 = attention_block(x3)  # Add attention
    x = concatenate([x, x3])
    
    # Global Average Pooling
    x = GlobalAveragePooling2D()(x)
    
    # Fully Connected Layer
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    # Output Layer
    outputs = Dense(15, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    return model

# Build and compile the model
model = build_snda()
model.compile(optimizer=Nadam(), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(datagen.flow(x_train, y_train, batch_size=128),
                    epochs=20,
                    validation_data=(x_val, y_val),
                    callbacks=[learning_rate_reduction])

# Evaluate the model
print("Accuracy of the model is - ", model.evaluate(x_test, y_test)[1] * 100, "%")



# Plot training and validation accuracy and loss
epochs = [i for i in range(20)]
fig, ax = plt.subplots(1, 2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
fig.set_size_inches(16, 9)

ax[0].plot(epochs, train_acc, 'go-', label='Training Accuracy')
ax[0].plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
ax[0].set_title('Training & Validation Accuracy')
ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")

ax[1].plot(epochs, train_loss, 'g-o', label='Training Loss')
ax[1].plot(epochs, val_loss, 'r-o', label='Validation Loss')
ax[1].set_title('Training & Validation Loss')
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Loss")
plt.show()
"""
# Generate predictions
predictions = np.argmax(model.predict(x_test), axis=1)
for i in range(len(predictions)):
    if predictions[i] >= 9:
        predictions[i] += 1

# Classification report and confusion matrix
classes = ["Class " + str(i) for i in range(25) if i != 9]
print(classification_report(y, predictions, target_names=classes))

cm = confusion_matrix(y, predictions)
cm = pd.DataFrame(cm, index=[i for i in range(25) if i != 9], columns=[i for i in range(25) if i != 9])
plt.figure(figsize=(15, 15))
sns.heatmap(cm, cmap="Blues", linecolor='black', linewidth=1, annot=True, fmt='')
plt.show()
"""
