import tensorflow as tf
from tensorflow.keras import layers, models

def build_improved_cnn_rnn(input_shape, num_classes):
    """
    input_shape: (sequence_length, height, width, channels)
    Example: (20, 48, 48, 1)
    """

    model = models.Sequential()

    # =========================
    # CNN FEATURE EXTRACTOR
    # applied per frame
    # =========================

    model.add(layers.TimeDistributed(
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        input_shape=input_shape
    ))
    model.add(layers.TimeDistributed(layers.BatchNormalization()))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((2, 2))))

    model.add(layers.TimeDistributed(
        layers.Conv2D(64, (3, 3), padding='same', activation='relu')
    ))
    model.add(layers.TimeDistributed(layers.BatchNormalization()))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((2, 2))))

    model.add(layers.TimeDistributed(
        layers.Conv2D(128, (3, 3), padding='same', activation='relu')
    ))
    model.add(layers.TimeDistributed(layers.BatchNormalization()))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((2, 2))))

    # =========================
    # KEY IMPROVEMENT:
    # Replace Flatten() with GlobalAveragePooling2D
    # =========================
    model.add(layers.TimeDistributed(
        layers.GlobalAveragePooling2D()
    ))

    # Dropout for regularization
    model.add(layers.TimeDistributed(layers.Dropout(0.4)))

    # =========================
    # TEMPORAL MODELING (GRU)
    # =========================

    model.add(layers.GRU(128, return_sequences=False))
    model.add(layers.Dropout(0.5))

    # =========================
    # CLASSIFIER
    # =========================

    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(num_classes, activation='softmax'))

    return model


# =========================
# PARAMETERS
# =========================

sequence_length = 20
img_height, img_width = 48, 48
channels = 1
num_classes = 7   # or 4 depending on your setup

model = build_improved_cnn_rnn(
    (sequence_length, img_height, img_width, channels),
    num_classes
)

# =========================
# COMPILATION
# =========================

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
