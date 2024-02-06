

import onnx   
import numpy as np
import tensorflow as tf
from onnx2tf.onnx2tf import convert 


def representative_dataset_gen():
    # Representative dataset generator for use with converter.representative_dataset, returns a generator of np arrays
    for _ in range(5):
        im = np.random.randn(1, 256, 256, 3)
        yield [im.astype(np.float32)] 



def rename_layers(onnx_model_path, renamed_model_path):
    # Load the original ONNX model
    original_model = onnx.load(onnx_model_path)

    # Define a function to sanitize the layer names
    def sanitize_name(name):
        sanitized_name = ""
        for char in name:
            if char.isalnum() or char in "._\\/>-":
                sanitized_name += char
            else:
                sanitized_name += "_"
        return sanitized_name

    # Rename the layers
    for i in range(len(original_model.graph.node)):
        original_model.graph.node[i].name = sanitize_name(original_model.graph.node[i].name)

    onnx.save(original_model, renamed_model_path)


# 1. Fix Ultralytics Naming of the Layers
rename_layers("path/to/onnx", "path/to/new-onnx") 

# 2. Convert to Keras  
tmp_model = convert("path/to/new-onnx")
converter = tf.lite.TFLiteConverter.from_keras_model(tmp_model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.int8]
converter.representative_dataset = representative_dataset_gen
converter.inference_type = tf.int8
converter.inference_input_type = tf.int8  # or tf.uint8
converter.inference_output_type = tf.int8  # or tf.uint8
tflite_model = converter.convert()
with open("path/to/tflite", "wb") as f: f.write(tflite_model)
