# Batch extract the undistorted_camera_parameters.zip 

import os
import shutil
import zipfile

root_path = '/home/w61/VLN/dataset/v1/scans'
backup_path = '/home/w61/VLN/dataset_copy/v1/scans' # just for back-up
fileList = os.listdir(root_path)

def unzip_file(zip_src, dst_dir):
    r = zipfile.is_zipfile(zip_src)
    if r:     
        fz = zipfile.ZipFile(zip_src, 'r')
        for file in fz.namelist():
            fz.extract(file, dst_dir)       
    else:
        print('This is not zip')

def extract_all_zip():
    count = 0
    for item in fileList:
        extract_dir = os.path.join(root_path,item,item)
        extract_target_dir = os.path.join(root_path,item)
        if not os.path.exists(os.path.join(extract_dir,'undistorted_camera_parameters')):
            zip_path = os.path.join(root_path,item,'undistorted_camera_parameters.zip')
            unzip_file(zip_path,extract_target_dir)
        if not os.path.exists(os.path.join(extract_dir,'matterport_skybox_images')):
            image_zip_path = os.path.join(root_path,item,'matterport_skybox_images.zip') 
            unzip_file(image_zip_path,extract_target_dir)
        print('extract {} succeesfully'.format(item))
        count += 1
    return count

def mv_all_dir():
    count = 0
    for item in fileList:
        extract_dir = os.path.join(root_path,item,item)
        target_dir = os.path.join(root_path,item)
        if os.path.exists(extract_dir):
            images_dir = os.path.join(extract_dir,'matterport_skybox_images')
            parameters_dir = os.path.join(extract_dir,'undistorted_camera_parameters')
            if not os.path.exists(os.path.join(target_dir,'matterport_skybox_images')):
                shutil.move(images_dir,target_dir)
            if not os.path.exists(os.path.join(target_dir,'undistorted_camera_parameters')):
                shutil.move(parameters_dir,target_dir)
                print('move {} successfully'.format(item))
                count += 1
    return count

def backup_all_zip():
    count = 0
    for item in fileList:
        current_dir = os.path.join(root_path,item)
        kid_fileList = os.listdir(current_dir)
        backup_kid_dir = os.path.join(backup_path,item)
        for file in kid_fileList:
            if os.path.splitext(file)[1] == '.zip':
                source_path = os.path.join(current_dir,file)
                if not os.path.exists(backup_kid_dir):
                    os.mkdir(os.path.join(backup_kid_dir))
                target_path = os.path.join(backup_kid_dir,file)
                shutil.move(source_path,target_path)
            count += 1
    return count

# count = mv_all_dir()
# item = '1LXtFkjw3qL'
# current_dir = os.path.join(root_path,item)
# kid_fileList = os.listdir(current_dir)
# backup_kid_dir = os.path.join(backup_path,item)
# for file in kid_fileList:
#     if os.path.splitext(file)[1] == '.zip':
#         source_path = os.path.join(current_dir,file)
#         if not os.path.exists(backup_kid_dir):
#             os.mkdir(os.path.join(backup_kid_dir))
#         target_path = os.path.join(backup_kid_dir,file)
#         shutil.move(source_path,target_path)

count = backup_all_zip()
print('total executed number:',count)

