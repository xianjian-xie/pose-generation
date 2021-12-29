import os
import shutil

from tqdm import tqdm

dataset_dir = 'C:\\Users\\Carl\\Desktop\\datasets\\Training_set_mask'
target_mask_dir = os.path.join(dataset_dir, 'mask')

if not os.path.exists(target_mask_dir):
    os.mkdir(target_mask_dir)

for subject_folder in tqdm([subject_folder for subject_folder in os.listdir(dataset_dir) if subject_folder != 'mask']):
    # print(f'Subject {subject_folder}')
    subject_dir = os.path.join(dataset_dir, subject_folder)
    for pose_folder in tqdm(os.listdir(subject_dir)):
        pose_dir = os.path.join(subject_dir, pose_folder)
        mask_dir = os.path.join(pose_dir, 'mask')

        # print('Moving images...')
        for img_file in os.listdir(mask_dir):
            img_path = os.path.join(mask_dir, img_file)
            shutil.move(img_path, target_mask_dir)
