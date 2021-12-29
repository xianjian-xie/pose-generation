import os
import shutil

from tqdm import tqdm

dataset_dir = 'datasets\\Training_set'
target_img_dir = os.path.join(dataset_dir, 'image')
target_kp_dir = os.path.join(dataset_dir, 'keypoints')

if not os.path.exists(target_img_dir):
    os.mkdir(target_img_dir)
if not os.path.exists(target_kp_dir):
    os.mkdir(target_kp_dir)

for subject_folder in tqdm([subject_folder for subject_folder in os.listdir(dataset_dir) if subject_folder != 'image' and subject_folder != 'keypoints']):
    # print(f'Subject {subject_folder}')
    subject_dir = os.path.join(dataset_dir, subject_folder)
    for pose_folder in tqdm(os.listdir(subject_dir)):
        pose_dir = os.path.join(subject_dir, pose_folder)
        img_dir = os.path.join(pose_dir, 'image')
        keypoint_dir = os.path.join(pose_dir, 'keypoints')

        # print('Moving images...')
        for img_file in os.listdir(img_dir):
            img_path = os.path.join(img_dir, img_file)
            shutil.move(img_path, target_img_dir)

        # print('Moving keypoints...')
        for keypoint_file in os.listdir(keypoint_dir):
            keypoint_path = os.path.join(keypoint_dir, keypoint_file)
            shutil.move(keypoint_path, target_kp_dir)
