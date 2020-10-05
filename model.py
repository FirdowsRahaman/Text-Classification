import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


VOCAB_SIZE = 2**13
EMBEDDING_DIMS = 64
NUM_CATEGORY = 4


def cnn_model():
  title_input = keras.Input(shape=(None,), name='title')
  x = layers.Embedding(VOCAB_SIZE, EMBEDDING_DIMS)(title_input)
  x = layers.Conv1D(128, 5, activation='relu')(title_features)
  x = layers.GlobalAveragePooling1D()(x)
  x = layers.Dense(64, activation='relu')(x)
  category_pred = layers.Dense(NUM_CATEGORY, name='category', activation='softmax')(x)
  model = keras.Model(inputs=[title_input], outputs=[category_pred])
  
  model.compile(loss='sparse_categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
  return model


def gru_model():
  title_input = keras.Input(shape=(None,), name='title')
  x = layers.Embedding(VOCAB_SIZE, EMBEDDING_DIMS)(title_input)
  x = layers.Bidirectional(layers.GRU(128)),(title_features)
  x = layers.Dense(64, activation='relu')(x)
  category_pred = layers.Dense(NUM_CATEGORY, name='category')(x)
  model = keras.Model(inputs=[title_input], outputs=[category_pred])
  
  model.compile(loss='sparse_categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
  return model