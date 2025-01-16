import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys
import os

IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
DATA_DIR = '/data/'
EPOCHS = 10
TRAIN_STOP = 0.93

def create_model():
    base_model = tf.keras.applications.DenseNet121(input_shape=(224, 224, 3), 
        include_top=False, 
        weights='imagenet')
    
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    return model

def load_data():
    script_dir = os.path.dirname(__file__)
    
    train_data_dir = script_dir + DATA_DIR + 'train'

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2  # Set validation split
    )

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training'
    )

    validation_data_dir = script_dir + DATA_DIR + 'valid'

    validation_generator = train_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation'
    )

    test_data_dir = script_dir + DATA_DIR + 'test'

    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    return train_generator, validation_generator, test_generator

def train_model(model, train_generator, validation_generator):

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if logs.get('accuracy') > TRAIN_STOP:
                self.model.stop_training = True

    callback = myCallback()

    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        callbacks=[callback],
        validation_data=validation_generator,
    )

    return history

def evaluate_model(model, test_generator):
    loss, accuracy = model.evaluate(test_generator)
    print(f"Validation Loss: {loss}")
    print(f"Validation Accuracy: {accuracy}")

def save_model(model):
    model.save('cutting_objects_detect_model.h5')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open('cutting_objects_detect_model.tflite', 'wb') as f:
        f.write(tflite_model)

def main():
    model = create_model()
    train_generator, validation_generator, test_generator = load_data()
    history = train_model(model, train_generator, validation_generator)
    evaluate_model(model, test_generator)
    save_model(model)

if __name__ == '__main__':
    main()