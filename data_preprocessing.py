import os
import numpy as np
from matplotlib.image import imread

def parse_training_data(dir_path):
    image_list, label_list =  [], []
    single_path_image_list, single_path_label_list = [], []
    for file in sorted(os.listdir(dir_path), key=lambda x: int(x.split("_")[1])):
        if file.endswith(".png"):
            print(file)
            file_path = os.path.join(dir_path, file)
            img = imread(file_path)
            single_path_image_list.append(img)
            single_path_label_list.append(file.split("_")[2])
            if len(single_path_image_list)%10 == 0:
                image_list.append(single_path_image_list.copy())
                single_path_image_list = []
                label_list.append(single_path_label_list.copy())
                single_path_label_list = []

    return np.asarray(image_list), np.asarray(label_list)



if __name__ == "__main__":
    images, labels = parse_training_data("./movo_data")

    print(images.shape) # Sanity check
    print(labels.shape)
    np.save("./movo_data/images.npy", images)
    np.save("./movo_data/labels.npy", labels)

