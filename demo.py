import os
import cv2
import time
import random
import argparse
import numpy as np
import tensorflow as tf


# coco_id_mapping = {
#     1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
#     6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
#     11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
#     16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep", 21: "cow",
#     22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe", 27: "backpack",
#     28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase", 34: "frisbee",
#     35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite",
#     39: "baseball bat", 40: "baseball glove", 41: "skateboard", 42: "surfboard",
#     43: "tennis racket", 44: "bottle", 46: "wine glass", 47: "cup", 48: "fork",
#     49: "knife", 50: "spoon", 51: "bowl", 52: "banana", 53: "apple",
#     54: "sandwich", 55: "orange", 56: "broccoli", 57: "carrot", 58: "hot dog",
#     59: "pizza", 60: "donut", 61: "cake", 62: "chair", 63: "couch",
#     64: "potted plant", 65: "bed", 67: "dining table", 70: "toilet", 72: "tv",
#     73: "laptop", 74: "mouse", 75: "remote", 76: "keyboard", 77: "cell phone",
#     78: "microwave", 79: "oven", 80: "toaster", 81: "sink", 82: "refrigerator",
#     84: "book", 85: "clock", 86: "vase", 87: "scissors", 88: "teddy bear",
#     89: "hair drier", 90: "toothbrush",
# }

coco_id_mapping = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane", 6: "bus", 
    7: "train", 8: "truck", 9: "boat", 10: "traffic light", 11: "fire hydrant", 
    12: "stop sign", 13: "parking meter", 14: "bench", 15: "bird", 16: "cat", 17: "dog", 
    18: "horse", 19: "sheep", 20: "cow", 21: "elephant", 22: "bear", 23:"zebra", 
    24: "giraffe", 25: "backpack", 26: "umbrella", 27: "handbag", 28: "tie", 29: "suitcase", 
    30: "frisbee", 31: "skis", 32: "snowboard", 33: "sports ball", 34: "kite", 35: "baseball bat", 
    36: "baseball glove", 37: "skateboard", 38: "surfboard", 39: "tennis racket", 40: "bottle", 
    41: "wine glass", 42: "cup", 43: "fork", 44: "knife", 45: "spoon", 46: "bowl", 47: "banana", 
    48: "apple", 49: "sandwich", 50: "orange", 51: "broccoli", 52: "carrot", 53: "hot dog", 54: "pizza", 
    55: "donut", 56: "cake", 57: "chair", 58: "couch", 59: "potted plant", 60: "bed", 61: "dining table", 
    62: "toilet", 63: "tv", 64: "laptop", 65: "mouse", 66: "remote", 67: "keyboard", 68: "cell phone",
    69: "microwave", 70: "oven", 71: "toaster", 72: "sink", 73: "refrigerator", 74: "book", 75: "clock", 
    76: "vase", 77: "scissors", 78: "teddy bear", 79: "hair drier", 80: "toothbrush"
}


def random_color(seed=None):
    random.seed(seed % 32)
    levels = range(32, 256, 32)

    return tuple(random.choice(levels) for _ in range(3))


def draw(img, box, label, score, names_dict, color=None):
    c1 = (int(box[0]), int(box[1]))
    c2 = (int(box[2]), int(box[3]))
    
    if int(label) not in names_dict:
        return img
    label = names_dict[int(label)]
    text = label + ":{:.2f}".format(float(score))
    # score = names_dict[detection[4].float()]

    # color = random.choice(self._colors)
    img = cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] - t_size[1] - 4
    img = cv2.rectangle(img, c1, c2, color, -1)
    # cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
    img = cv2.putText(img, text, (c1[0], c1[1]), cv2.FONT_HERSHEY_PLAIN, 1, [0, 0, 0], 1)

    return img


def preprocess(image, input_scale=None):
    height, width, _ = image.shape
    if input_scale is None:
        new_size = (width // 64 * 64, height // 64 * 64)
        input_scale = new_size
        dw, dh = 0, 0
        ratio = new_size[0] / height, new_size[1] / width
    # ratio = height / width
    # if height < width:
    #     width = input_scale if input_scale else int((width // 32) * 32)
    #     height = int(width * ratio // 32 * 32)
    # else:
    #     height = input_scale if input_scale else int(height // 32 * 32)
    #     width = int(height / ratio // 32 * 32)
    else:
        if isinstance(input_scale, int):
            input_scale = [input_scale] * 2
        r = min(input_scale[0] / height, input_scale[1] / width)
        # r = min(r, 1.0)
        ratio = r, r
                
        new_size = int(round(r * width)), int(round(r * height))
        dw, dh = input_scale[1] - new_size[0], input_scale[0] - new_size[1]
        # print(dw, dh)
        # dw, dh = np.mod(dw, 32), np.mod(dh, 32)

    # image = cv2.resize(image, new_size)
    image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), new_size)
    image = cv2.copyMakeBorder(image, 0, dh, 0, dw, cv2.BORDER_CONSTANT, value=(0, 0, 0))  # add border

    return image, image.shape[0:2], (new_size[1], new_size[0]), ratio


def inference_img(saved_model_dir, img_path, input_size):
    loaded = tf.saved_model.load(saved_model_dir, tags=[tf.saved_model.SERVING])

    infer = loaded.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    infer = loaded.signatures["serving_default"]

    if not os.path.exists(img_path):
        raise ValueError("ERROR: %s not exists." % img_path)
    
    img = cv2.imread(img_path)
    img_data, input_size, valid_size, ratio = preprocess(img, input_size)
    
    img_data = tf.convert_to_tensor(img_data[None, ...], tf.uint8)
    # img_data = tf.tile(img_data, [2, 1, 1, 1])
  
    outputs = infer(img_data)

    if "valid_detections" in outputs:
        num = outputs["valid_detections"].numpy()[0]
        boxes = outputs["nmsed_boxes"].numpy()[0][:num]
        scores = outputs["nmsed_scores"].numpy()[0][:num]
        classes = outputs["nmsed_classes"].numpy()[0][:num]
    else:
        num = outputs["nms_3"].numpy()[0]
        boxes = outputs["nms"].numpy()[0][:num]
        scores = outputs["nms_1"].numpy()[0][:num]
        classes = outputs["nms_2"].numpy()[0][:num]

    for i in range(num):
        box = boxes[i] * np.array([input_size[0], input_size[1]] * 2)
        box = box / np.array([ratio[1], ratio[0], ratio[1], ratio[0]])
        
        # box = boxes[i] * np.array([height, width, height, width])
        cls = classes[i] + 1
        img = draw(img, box, cls, scores[i], coco_id_mapping, random_color(int(cls)))
 
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def inference(saved_model_dir, video_path, input_size):
    loaded = tf.saved_model.load(saved_model_dir, tags=[tf.saved_model.SERVING])

    infer = loaded.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    infer = loaded.signatures["serving_default"]

    if not os.path.exists(video_path):
        raise ValueError("ERROR: %s not exists." % video_path)

    video = cv2.VideoCapture(video_path)

    exec_time = 0.1 
    while True:
        return_value, frame = video.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("No image!")
        frame_size = frame.shape[:2]
        img_data, input_size, valid_size, ratio = preprocess(frame, input_size)
        
        prev_time = time.time()
        img_data = tf.convert_to_tensor(img_data[None, ...], tf.uint8)
        outputs = infer(img_data)
       
        num = outputs["valid_detections"].numpy()[0]
        boxes = outputs["nmsed_boxes"].numpy()[0][:num]
        scores = outputs["nmsed_scores"].numpy()[0][:num]
        classes = outputs["nmsed_classes"].numpy()[0][:num]
        for i in range(num):
            box = boxes[i] * np.array([input_size[0], input_size[1]] * 2)
            box = box / np.array([ratio[1], ratio[0], ratio[1], ratio[0]])
            # box = boxes[i] * np.array([height, width, height, width])
            cls = classes[i]
            frame = draw(frame, box, cls + 1, scores[i], coco_id_mapping, random_color(int(cls)))

        curr_time = time.time()
        exec_time = exec_time * 0.9 + 0.1 * (curr_time - prev_time)

        info = "time: %.2f ms" % (1000 * exec_time)
        cv2.putText(frame, text=info, org=(50, 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        result = cv2.resize(result, (1024, 576))
        cv2.imshow("result", result)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Demo args")
    parser.add_argument("--saved_model_dir", required=True, type=str, help="saved model directory")
    parser.add_argument("--img_path", default=None, type=str, help="image path")
    parser.add_argument("--video_path", default=None, type=str, help="video for demo.")
    parser.add_argument("--input_size", required=True, type=int, help="input size")

    args = parser.parse_args()
    
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device=device, enable=True)
        
    saved_model_dir = args.saved_model_dir
    video_path = args.video_path
    img_path = args.img_path
    if img_path is not None:
        inference_img(saved_model_dir, img_path, args.input_size)
    elif video_path is not None:
        inference(saved_model_dir, video_path, args.input_size)
    else:
        raise ValueError("ERROR: must provide image or video.")
        

if __name__ == "__main__":
    main()
