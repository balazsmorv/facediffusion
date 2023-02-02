import os
import shutil

source_folder = "/datadrive/facediffusion/dataset/train/images/"
destination_folder = "/datadrive/facediffusion/dataset/small_train/images/"

# fetch all files
i = 0
print("hello")
for file_name in os.listdir(source_folder):
    # construct full file path
    source = source_folder + file_name
    destination = destination_folder + file_name
    if i % 10 == 0:
        shutil.move(source, destination)
        print('Moved:', file_name)
    i += 1
