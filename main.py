from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import datetime

from model import cnn_model
from data_transform import load_data, text_transform


def train_and_evaluate(params):
    batch_size = params['batch_size']
    num_epochs = params['num_epochs']
    output_dir = params['output_dir']
    train_csv_file = params['train_dir']
    valid_csv_file = params['valid_dir']
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    savedmodel_dir = os.path.join(output_dir, 'savedmodel')
    model_export_path = os.path.join(savedmodel_dir, timestamp)
  
  if tf.io.gfile.exists(output_dir):
    tf.io.gfile.rmtree(output_dir)
    
    train_text, train_labels = load_data(train_csv_file)
    valid_text, valid_labels = load_data(valid_csv_file)
    
    train_ds = text_transform(train_text)
    valid_ds = text_transform(valid_text)
    
    model = cnn_model()
    history = model.fit(train_ds, train_labels
                        batch_size=batch_size,
                        epochs=num_epochs,
                        validation_data=(valid_ds, valid_labels))
    
    tf.saved_model.save(model, model_export_path)
    return history