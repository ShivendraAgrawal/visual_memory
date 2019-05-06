import os
import numpy as np
from matplotlib.image import imread

def parse_training_data(dir_path):
    label_transform = {'i':0, 'j':1, 'l':2}
    image_list, depth_list, label_list =  [], [], []
    single_path_image_list, single_path_depth_list,single_path_label_list = [], [], []
    files = [file for file in os.listdir(dir_path) if file.endswith(".png")]
    for file in sorted(files, key=lambda x: int(x.split("_")[1])):
        if file.endswith(".png"):
            print(file)
            file_path = os.path.join(dir_path, file)
            img = imread(file_path)
            depth_path = os.path.join(dir_path,
                                      str(file.split(".")[0]) + "." +
                                      str(file.split(".")[1]) + ".npy")
            depth = np.load(depth_path)
            single_path_depth_list.append(depth)
            single_path_image_list.append(img)
            single_path_label_list.append(label_transform[file.split("_")[2]])
            if len(single_path_image_list)%10 == 0:
                image_list.append(single_path_image_list.copy())
                single_path_image_list = []
                depth_list.append(single_path_depth_list.copy())
                single_path_depth_list = []
                label_list.append(single_path_label_list.copy())
                single_path_label_list = []

    return np.asarray(image_list), np.asarray(depth_list), np.asarray(label_list)



if __name__ == "__main__":
    images, depths, labels = parse_training_data("/Users/shivendra/Downloads/movo_data")

    print(images.shape) # Sanity check
    print(depths.shape)
    print(labels.shape)
    np.save("./movo_data/depths_v2.npy", depths)
    np.save("./movo_data/images_v2.npy", images)
    np.save("./movo_data/labels_v2.npy", labels)

