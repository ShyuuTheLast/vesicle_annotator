import logging
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import matplotlib

from scripts.data_loader import create_data_loader
from scripts.vesicle_net import VesicleNet, create_model, load_checkpoint, train_model
from scripts.visualization import html_visualize

def train_and_save_model(image_file, mask_file, label_file, checkpoint_path, batch_size, n_channels, n_classes,
                         num_epochs, lr, momentum):
    # Create the data loader
    train_loader = create_data_loader(image_file, mask_file, label_file, batch_size)

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
    val_loader = create_data_loader(image_file, mask_file, label_file, batch_size, shuffle=False)

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

def show_visuals(image_file, mask_file, label_file, checkpoint_path, save_dir, n_channels, n_classes, batch_size, lr=0.001,
                 momentum=0.9):
    logging.info("Custom Dataset Processing")

    logging.info('==> Evaluating ...')

    start_time = time.time()

    # Create data loader for test images
    data_loader = create_data_loader(image_file, mask_file, label_file, batch_size, shuffle=False)

    # Initialize the model and optimizer
    model = VesicleNet(in_channels=n_channels, num_classes=n_classes)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # Load model checkpoint
    load_checkpoint(model, optimizer, checkpoint_path)

    # Perform predictions on test images and generate visualizations
    html_visualize(model, data_loader, save_dir, n_classes)

    end_time = time.time()
    print(f"Visualizations generated in {end_time - start_time:.2f} seconds")


if __name__ == '__main__':

    # Initialize file paths
    train_image_file = 'data/big_vesicle_cls/bigV_cls_im_v2.h5'
    train_mask_file = 'data/big_vesicle_cls/bigV_cls_mask_v2.h5'
    train_label_file = 'data/big_vesicle_cls/bigV_cls_label_v2.h5'

    test_image_file = 'data/big_vesicle_cls_testing/bigV_cls_202406_im.h5'
    test_mask_file = 'data/big_vesicle_cls_testing/bigV_cls_202406_mask.h5'
    test_label_file = None

    checkpoint_path = 'model_checkpoint.pth'
    visualization_save_dir = 'testing_predictions'

    # Model and training parameters
    num_epochs = 40
    batch_size = 128
    n_channels = 1
    n_classes = 3
    lr = 0.001
    momentum = 0.9

    matplotlib.use('Agg')

    #train_and_save_model(train_image_file, train_mask_file, train_label_file, checkpoint_path, batch_size, n_channels, n_classes, num_epochs, lr, momentum)

    #eval_model_results(train_image_file, train_mask_file, train_label_file, checkpoint_path, n_channels, n_classes, batch_size, lr, momentum)

    show_visuals(test_image_file, test_mask_file, test_label_file, checkpoint_path, visualization_save_dir, n_channels, n_classes, batch_size, lr, momentum)


