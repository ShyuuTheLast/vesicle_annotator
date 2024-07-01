import glob
import time

from scripts.getting_started_script import create_model, train_model, create_data_loaders, BATCH_SIZE
from scripts.create_visuals import evaluate_and_visualize_model_pyvista


def delete_all_obj_files(project_root):
    obj_files = glob.glob(os.path.join(project_root, '*.obj'))
    for file in obj_files:
        try:
            os.remove(file)
            print(f"Deleted: {file}")
        except Exception as e:
            print(f"Error deleting file {file}: {e}")

import os
import shutil

def delete_batch_folders(root_dir):
    """
    Deletes all folders in the specified root directory that start with "batch".

    Parameters:
    root_dir (str): The root directory to search for folders starting with "batch".
    """
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        if os.path.isdir(folder_path) and folder_name.startswith("batch"):
            try:
                shutil.rmtree(folder_path)
                print(f"Deleted folder: {folder_path}")
            except Exception as e:
                print(f"Error deleting folder {folder_path}: {e}")

# Example usage
root_directory = '/Users/ayalyakobe/vesicle_annotations'
delete_batch_folders(root_directory)


if __name__ == '__main__':
    start_time = time.time()  # Start the timer

    print("Custom Dataset Processing")
    image_file = '/Users/ayalyakobe/vesicle_annotations/big_vesicle_cls/bigV_cls_im.h5'
    label_file = '/Users/ayalyakobe/vesicle_annotations/big_vesicle_cls/bigV_cls_label.h5'
    mask_file = '/Users/ayalyakobe/vesicle_annotations/big_vesicle_cls/bigV_cls_mask.h5'
    task = 'classification'  # Specify the task type
    n_channels = 1  # Number of channels for grayscale image_predictions
    n_classes = 3  # Number of classes

    train_loader = create_data_loaders(image_file, label_file, mask_file, BATCH_SIZE)

    model, criterion, optimizer = create_model(n_channels, n_classes, task)
    train_model(model, criterion, optimizer, train_loader, task)

    end_time = time.time()  # End the timer
    elapsed_time = end_time - start_time
    print(f'Total time taken: {elapsed_time:.2f} seconds')

    print('==> Evaluating ...')
    # Evaluate and visualize the model
    # evaluate_and_visualize_model_3d(model, train_loader, save_dir)
    evaluate_and_visualize_model_pyvista(model, train_loader, '/image_predictions/whole_predictions')

