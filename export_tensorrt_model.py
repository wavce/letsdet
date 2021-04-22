import os
import glob
import argparse
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt


def calibrating_dataset(image_dir, batch_size, input_size):
    image_names = glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.png"))
    image_names = tf.convert_to_tensor(image_names)
    dataset = tf.data.Dataset.from_tensor_slices(image_names)

    def _fn(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3)
        image = tf.image.resize(image, [input_size, input_size])

        image = tf.cast(image, tf.uint8)

        return (image, )

    dataset = dataset.map(map_func=_fn, num_parallel_calls=8)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(1)

    return dataset


# Define a generator function that yields input data, and use it to execute
# the graph to build TRT engines.
# With TensorRT 5.1, different engines will be built (and saved later) for
# different input shapes to the TRTEngineOp.
def my_input_fn(input_size):
    for _ in range(100):
        value = tf.random.uniform(
            [1, input_size, input_size, 3], 0, 255, tf.float32)
        yield (tf.cast(value, tf.uint8), )


parser = argparse.ArgumentParser()
parser.add_argument("--saved_model_dir", required=True, type=str)
parser.add_argument("--export_dir", required=True, type=str)
parser.add_argument("--mode", default="FP16", type=str)
parser.add_argument("--image_dir", default="", type=str)
parser.add_argument("--input_size", default=224, type=int)
parser.add_argument("--max_workspace_size_bytes", default=2<<20, type=int, 
                    help="The maximum GPU temporary memory which the TRT engine can use at"
                         " execution time. This corresponds to the `workspaceSize` parameter of"
                         " nvinfer1::IBuilder::setMaxWorkspaceSize().")
parser.add_argument("--minimum_segment_size", default=3, type=int, 
  	                help="The minimum number of nodes required for a subgraph to be replaced by TRTEnginOp.")
parser.add_argument("--maximum_cached_engines", default=100, type=int, 
                    help="Max number of cached TRT engines in dynamic TRT ops. If the number of cached engines"
                         " is already at max but nonr them can serve the input, the TRTEngineOp will fall back"
                         " to run the TF function based on which the TRTEngineOp is created.")
parser.add_argument("--optimization_offline", action="store_true", help="")


args = parser.parse_args()
print(args)

# assert args.mode in ["FP16", "FP32"], "but only support `FP16` and `FP32`."

if args.mode in ("FP16", "FP32"):
    # FP32/FP16 precision with pre-built engines
    params = trt.DEFAULT_TRT_CONVERSION_PARAMS
    params._replace(precision_mode=args.mode)
    # Set this to a large enough number so it can cache all the engines.
    params._replace(maximum_cached_engines=args.maximum_cached_engines)
    params._replace(minimum_segment_size=args.minimum_segment_size)
    params._replace(use_calibration=False)
    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=args.saved_model_dir, 
        conversion_params=params)
    converter.convert()

else:
    assert args.mode == "INT8"
    params = trt.DEFAULT_TRT_CONVERSION_PARAMS
    params._replace(precision_mode=args.mode)
    # Set this to a large enough number so it can cache all the engines.
    params._replace(maximum_cached_engines=args.maximum_cached_engines)
    params._replace(minimum_segment_size=args.minimum_segment_size)
    params._replace(use_calibration=True)
    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=args.saved_model_dir, 
        conversion_params=params)

    # Define a generator function that yields input data, and run INT8
    # calibration with the data. All input data should have the same shape.
    # At the end of convert(), the calibration stats (e.g. range information)
    # will be saved and can be used to generate more TRT engines with different
    # shapes. Also, one TRT engine will be generated (with the same shape as
    # the calibration data) for save later.
    def my_calibration_input_fn():
        dataset = calibrating_dataset(args.image_dir, 16, args.input_size)
        for image_batch in dataset.take(1000):
            
            yield (image_batch, )

    converter.convert(calibration_input_fn=my_calibration_input_fn)

if args.optimization_offline:
    print("optimize offline ...")
    converter.build(input_fn=lambda: my_input_fn(args.input_size))  # Generate corresponding TRT engines

export_dir = args.export_dir or "./trt_models/1"
converter.save(export_dir)  # Generated engines will be saved.

print("Saved model to", export_dir)

