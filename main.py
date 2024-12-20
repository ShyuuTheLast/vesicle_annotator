import logging
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import matplotlib
import re
import glob
import os
import shutil
import h5py

from scripts.html_visualization import HtmlGenerator
from scripts.data_loader import create_data_loader
from scripts.vesicle_net import VesicleNet, create_model, load_checkpoint, train_model
from scripts.img_visualization import generate_images


def sort_files_to_directories(source_folder):
    """
    Sorts files in the source_folder into subdirectories based on the neuron name in the file names.
    Prints the list of subfolders created.

    Args:
        source_folder (str): Path to the folder containing files to be sorted.

    Returns:
        None
    """
    if not os.path.exists(source_folder):
        raise FileNotFoundError(f"Source folder does not exist: {source_folder}")

    # Get all files in the source folder
    files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

    # Set to track unique subfolder names
    subfolders_created = set()

    for file_name in files:
        # Extract the neuron name from the file name (assuming format like "vesicle_big_KR4_30-8-8_patch.h5")
        parts = file_name.split('_')  # Split by underscores
        neuron_name = parts[2]  # Assume neuron name is always the third part

        # Define the destination directory for this neuron
        neuron_dir = os.path.join(os.path.dirname(source_folder), neuron_name)

        # Add the subfolder name to the set
        subfolders_created.add(neuron_name)

        # Create the directory if it doesn't exist
        os.makedirs(neuron_dir, exist_ok=True)

        # Move the file to the neuron directory
        source_path = os.path.join(source_folder, file_name)
        dest_path = os.path.join(neuron_dir, file_name)
        shutil.move(source_path, dest_path)

        print(f"Moved {file_name} to {neuron_dir}")

    # Print the list of subfolders created
    return list(subfolders_created)


def split_h5_file(file_path):
    """
    Splits a single HDF5 file containing two datasets into two separate files.
    The first dataset is saved with '_im' in the name, and the second with '_mask'.

    Args:
        file_path (str): Path to the input HDF5 file.

    Returns:
        tuple: Paths to the generated '_im' and '_mask' HDF5 files.
    """
    # Extract the base file name without extension
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    # Define output file names
    im_file_path = os.path.join(os.path.dirname(file_path), "im.h5")
    mask_file_path = os.path.join(os.path.dirname(file_path), "mask.h5")

    # Open the input HDF5 file and separate datasets
    with h5py.File(file_path, 'r') as input_file:
        dataset_names = list(input_file.keys())

        if len(dataset_names) != 2:
            raise ValueError(f"Expected exactly 2 datasets, but found {len(dataset_names)}.")

        # Save the first dataset to a new file with "_im" in the name
        with h5py.File(im_file_path, 'w') as im_file:
            im_file.create_dataset("main", data=input_file[dataset_names[0]][:])
        print(f"First dataset saved to: {im_file_path}")

        # Save the second dataset to a new file with "_mask" in the name
        with h5py.File(mask_file_path, 'w') as mask_file:
            mask_file.create_dataset("main", data=input_file[dataset_names[1]][:])
        print(f"Second dataset saved to: {mask_file_path}")

    return im_file_path, mask_file_path


def train_and_save_model(image_file, mask_file, label_file, checkpoint_path, batch_size, n_channels, n_classes,
                         num_epochs, lr, momentum):
    # Create the data loader
    train_loader = create_data_loader(image_file, mask_file, label_file, None, batch_size)

    # Initialize the model, criterion, and optimizer with learning rate and momentum
    model, criterion, optimizer = create_model(n_channels, n_classes, lr, momentum)

    # Load the checkpoint if it exists
    start_epoch = load_checkpoint(model, optimizer, checkpoint_path)

    # Train the model
    start_time = time.time()
    train_model(model, criterion, optimizer, train_loader, start_epoch, num_epochs, checkpoint_path)
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")


def eval_model_results(image_file, mask_file, label_file, checkpoint_path, n_channels, n_classes, batch_size, lr=0.001,
                       momentum=0.9):
    # Initialize the model and optimizer
    model = VesicleNet(in_channels=n_channels, num_classes=n_classes)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # Load the model checkpoint
    load_checkpoint(model, optimizer, checkpoint_path)
    print("Evaluating model on validation/test data")

    # Create the data loader for validation/test data without shuffling
    val_loader = create_data_loader(image_file, mask_file, label_file, None, batch_size, shuffle=False)

    y_true = []
    y_pred = []

    model.eval()
    with torch.no_grad():
        for inputs, masks, labels in val_loader:  # Adjusted to match the correct return order
            inputs = inputs.float().unsqueeze(1)  # Adding channel dimension for grayscale
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy().flatten())
            y_pred.extend(predicted.cpu().numpy().flatten())

    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    # Print evaluation results
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation Precision: {precision:.4f}")
    print(f"Validation Recall: {recall:.4f}")
    print(f"Validation F1 Score: {f1:.4f}")


def extract_number(string):
    numbers = re.findall(r'\d+', string)  # Find all numeric substrings
    return int(numbers[-1]) if numbers else -1  # Return the last number if found, otherwise return -1


def get_image_paths_from_folder(folder_dir):
    """
    Retrieve image paths relative to the html_files directory, assuming a fixed folder structure.

    Args:
        folder_dir (str): The directory containing the images (e.g., category_a or category_b).

    Returns:
        list: A list of lists, where each sublist contains a relative path to an image.
    """
    # Get all .png files in the folder and subfolders
    image_paths = glob.glob(os.path.join(folder_dir, '**', '*.png'), recursive=True)

    # Sort image paths numerically based on the number in the filename
    image_paths.sort(key=extract_number)

    # Prepend '../' and construct the relative paths
    category_name = os.path.basename(folder_dir)  # Get the name of the category (e.g., 'category_a')
    relative_paths = [
        [f"../{category_name}/{os.path.basename(path)}"]
        for path in image_paths
    ]

    return relative_paths


def generate_html(input_folder, output_folder, color_labels):
    # Get all subfolders (CV, DV, DVH)
    subfolders = [f.path for f in os.scandir(input_folder) if f.is_dir() and os.path.basename(f.path) in color_labels]

    num_user = 1  # number of users

    all_image_paths = []
    all_image_labels = []

    # Loop through each subfolder (CV, DV, DVH)
    for subfolder in subfolders:
        # Get the name of the subfolder (e.g., CV, DV, DVH)
        category = os.path.basename(subfolder)

        # Generate the image paths from the subfolder
        image_paths = get_image_paths_from_folder(subfolder)
        all_image_paths.append(image_paths)

        # Construct the HDF5 label file name with category
        label_file = os.path.join(subfolder, f'{category}.h5')

        # Open the HDF5 file to get image labels
        dataset_name = "main"
        with h5py.File(label_file, 'r') as f:
            image_labels = np.array(f[dataset_name])

        # Append the image labels for this subfolder to the list of all image labels
        all_image_labels.append(image_labels)

    # Generate HTML for this category using HtmlGenerator
    html = HtmlGenerator(input_folder, output_folder, subfolders, color_labels, num_user=num_user, num_column=2)
    html.create_html(all_image_paths, all_image_labels)


def predict_images(image_file, mask_file, label_file, bounding_box_file, checkpoint_path, save_dir, n_channels,
                   n_classes, batch_size, lr=0.001,
                   momentum=0.9):
    logging.info("Custom Dataset Processing")
    logging.info('==> Evaluating ...')

    start_time = time.time()

    # Create data loader for test images
    data_loader = create_data_loader(image_file, mask_file, label_file, bounding_box_file, batch_size, shuffle=False)
    # Initialize the model and optimizer
    model = VesicleNet(in_channels=n_channels, num_classes=n_classes)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # Load model checkpoint
    load_checkpoint(model, optimizer, checkpoint_path)
    # Perform predictions on test images and generate visualizations
    generate_images(model, data_loader, save_dir)

    end_time = time.time()
    print(f"Visualizations generated in {end_time - start_time:.2f} seconds")


if __name__ == '__main__':

    checkpoint_path = 'model_checkpoint.pth'
    color_labels = ["undefined", "CV", "DV", "DVH"]
    # Model and training parameters
    num_epochs = 40
    batch_size = 128
    n_channels = 1
    n_classes = 3
    lr = 0.001
    momentum = 0.9
    matplotlib.use('Agg')

    # Initialize file paths
    train_image_file = 'data/big_vesicle_cls/bigV_cls_im_v2.h5'
    train_mask_file = 'data/big_vesicle_cls/bigV_cls_mask_v2.h5'
    train_label_file = 'data/big_vesicle_cls/bigV_cls_label_v2.h5'

    #train_and_save_model(train_image_file, train_mask_file, train_label_file, checkpoint_path, batch_size, n_channels, n_classes, num_epochs, lr, momentum)

    #eval_model_results(train_image_file, train_mask_file, train_label_file, checkpoint_path, n_channels, n_classes, batch_size, lr, momentum)

    visualize_training = False
    if visualize_training:
        visualization_save_dir = 'training_predictions'
        predict_images(train_image_file, train_mask_file, train_label_file, checkpoint_path, visualization_save_dir,
                       n_channels, n_classes, batch_size, lr, momentum)
        generate_html(visualization_save_dir, visualization_save_dir, color_labels)

    #test_image_file = 'data/big_vesicle_cls_testing/bigV_cls_202406_im.h5'
    #test_mask_file = 'data/big_vesicle_cls_testing/bigV_cls_202406_mask.h5'

    source_folder = "data/to_be_sorted"
    neuron_list = sort_files_to_directories(source_folder)
    #resut_sub_folders = [os.path.basename(f.path) for f in os.scandir("results")]
    #neuron_list = resut_sub_folders

    for neuron_name in neuron_list:
        print(f"Beginning visualization for {neuron_name}...")
        data_dir = os.path.join("data", neuron_name)

        visualize_testing = True
        if visualize_testing:
            # Define the input file paths based on the neuron name
            vesicle_file = os.path.join(data_dir, f"vesicle_big_{neuron_name}_30-8-8_patch.h5")
            bounding_box_file = os.path.join(data_dir, f"vesicle_big-bbs_{neuron_name}_30-8-8.h5")

            # Split the vesicle HDF5 file
            test_image_file, test_mask_file = split_h5_file(vesicle_file)

            # No test label file for this step
            test_label_file = None

            save_dir = os.path.join("results", neuron_name)

            # Run prediction and HTML generation
            predict_images(
                test_image_file,
                test_mask_file,
                test_label_file,
                bounding_box_file,
                checkpoint_path,
                save_dir,
                n_channels,
                n_classes,
                batch_size,
                lr,
                momentum
            )

            generate_html(save_dir, save_dir, color_labels)
        print(f"{neuron_name} has been visualized.")