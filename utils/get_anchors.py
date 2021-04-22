import argparse
import numpy as np


def iou(box, clusters):
    """
    Calculates the Intersection over Union (IoU) between a box and k clusters.
    :param box: tuple or array, shifted to the origin (i. e. width and height)
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: numpy array of shape (k, 0) where k is the number of clusters
    """
    #计算每个box与9个clusters的iou
    # boxes ： 所有的[[width, height], [width, height], …… ]
    # clusters : 9个随机的中心点[width, height]
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")
 
    intersection = x * y
    # 所有的boxes的面积
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]
 
    iou_ = intersection / (box_area + cluster_area - intersection)
 
    return iou_
 
 
def avg_iou(boxes, clusters):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: average IoU as a single float
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])
 
 
def translate_boxes(boxes):
    """
    Translates all the boxes to the origin.
    :param boxes: numpy array of shape (r, 4)
    :return: numpy array of shape (r, 2)
    """
    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
    return np.delete(new_boxes, [0, 1], axis=1)
 
 
def kmeans(boxes, k, dist=np.median):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param k: number of clusters
    :param dist: distance function
    :return: numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]
    distances = np.empty((rows, k))
 
    last_clusters = np.zeros((rows,))
    np.random.seed()
    # the Forgy method will fail if the whole array contains the same rows
    #初始化k个聚类中心（从原始数据集中随机选择k个）
    clusters = boxes[np.random.choice(rows, k, replace=False)]
    while True:
        for row in range(rows):
            # 定义的距离度量公式：d(box,centroid)=1-IOU(box,centroid)。到聚类中心的距离越小越好，
            # 但IOU值是越大越好，所以使用 1 - IOU，这样就保证距离越小，IOU值越大。
            # 计算所有的boxes和clusters的值（row，k）
            distances[row] = 1 - iou(boxes[row], clusters)
            #print(distances)
        # 将标注框分配给“距离”最近的聚类中心（也就是这里代码就是选出（对于每一个box）距离最小的那个聚类中心）。
        nearest_clusters = np.argmin(distances, axis=1)
        # 直到聚类中心改变量为0（也就是聚类中心不变了）。
        if (last_clusters == nearest_clusters).all():
            break
        # 计算每个群的中心（这里把每一个类的中位数作为新的聚类中心）
        for cluster in range(k):
            #这一句是把所有的boxes分到k堆数据中,比较别扭，就是分好了k堆数据，每堆求它的中位数作为新的点
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)
        last_clusters = nearest_clusters

    return clusters


def get_dataset(path):
    dataset = []
    with open(path, "r") as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        line = line.replace("jpg", "txt")

        with open(line, "r") as f:
            anns = f.readlines()
        
        for ann in anns:
            ann = ann.strip()
            _, x, y, w, h = ann.split(" ")
        
            dataset.append([float(w), float(h)])
    
    return np.array(dataset)


parser = argparse.ArgumentParser(description="The arguments for getting anchors using Kmeans in YOLO")
parser.add_argument("--input_scale", type=int, default=416, help="The input scale.")
parser.add_argument("--data_path", type=str, required=True, help="The txt file for all data, e.g. train.txt")

args = parser.parse_args()

INPUT_SCALE = args.input_scale
data_path = args.data_path

data = get_dataset(data_path)
data = data[data[:, 0] > 0]
data = data[data[:, 1] > 0]
print("Total boxs:", len(data))
out = kmeans(data, 9)
print("Anchors:", out* INPUT_SCALE)
print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))
#print("Boxes:\n {}".format(out)) 
ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
print("Ratios:\n {}".format(sorted(ratios)))

anchors = out * INPUT_SCALE
areas = anchors[:, 0] * anchors[:, 1]
indices = np.argsort(areas)
anchors = anchors[indices]
print(anchors)


with open("./anchors.txt", "w") as f:
    for (w, h) in anchors:
        f.write("%d %d " % (w, h))
