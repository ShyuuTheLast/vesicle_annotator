import os
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import find_contours

# Function to visualize image, prediction, and mask
def get_label_text(label):
    if label == 0:
        return 'CV'
    elif label == 1:
        return 'DV'
    elif label == 2:
        return 'DVH'
    else:
        print(f"Warning: Unrecognized label '{label}'. Defaulting to 'Unknown'.")
        return 'Unknown'

def visualize_image_with_prediction(image, mask, label, pred_label, save_dir, num_images_saved):
    num_slices = image.shape[1]  # Number of slices in the second dimension
    fig, ax = plt.subplots(1, num_slices, figsize=(20, 6))

    for i in range(num_slices):
        contours_mask = find_contours(mask[i], level=0.5)

        pred_label_text = get_label_text(pred_label)
        if label != -1:
            true_label_text = get_label_text(label)

        # Display the original image slice with mask outline
        ax[i].imshow(image[:, i, :], cmap='gray')
        for contour in contours_mask:
            ax[i].plot(contour[:, 1], contour[:, 0], linewidth=2, color='red')
        ax[i].set_title(f'Slice {i}')
        ax[i].axis('off')

    if label != -1:
        plt.suptitle(f'ID: {num_images_saved}, True Label: {true_label_text}, Predicted Label: {pred_label_text}', fontsize=18)
    else:
        plt.suptitle(f'ID: {num_images_saved}, Predicted Label: {pred_label_text}', fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    category_folder = os.path.join(save_dir, pred_label_text)
    os.makedirs(category_folder, exist_ok=True)

    # Save the figure
    save_path = os.path.join(category_folder, f'{num_images_saved}.png')
    plt.savefig(save_path)
    plt.close(fig)

def generate_images(model, data_loader, save_dir):
    """
    Evaluates the model on the provided data_loader, generates predictions, and saves visualizations.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        data_loader (torch.utils.data.DataLoader): DataLoader providing the test data.
        save_dir (str): Directory where the visualizations and HTML will be saved.
    """
    model.eval()
    num_images_saved = 0  # Counter to keep track of saved images
    image_paths = []  # List to store paths of saved images
    os.makedirs(save_dir, exist_ok=True)

    category_predictions = {}

    with torch.no_grad():
        for inputs, masks, label in data_loader:  # Adjusted to match the correct return order
            print(f"Processing batch of {inputs.size(0)} images.")
            inputs = inputs.float().unsqueeze(1)  # Adding channel dimension for grayscale
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            for i in range(inputs.size(0)):


                image = inputs[i, 0].cpu().numpy()
                prediction = predicted[i].cpu().numpy()
                mask = masks[i].cpu().numpy()
                pred_label = predicted[i].cpu().item()
                true_label = label[i]

                category_name = get_label_text(pred_label)
                if category_name not in category_predictions:
                    category_predictions[category_name] = []
                category_predictions[category_name].append(prediction + 1)

                # Visualize the prediction and save the image
                visualize_image_with_prediction(image, mask, true_label, pred_label, save_dir, num_images_saved)

                num_images_saved += 1  # Increment the counter

                if num_images_saved % 64 == 0:
                    print(f"Checkpoint: {num_images_saved} images processed")

    for category, predictions in category_predictions.items():
        category_folder = os.path.join(save_dir, category)
        os.makedirs(category_folder, exist_ok=True)
        category_file_path = os.path.join(category_folder, f'{category}.h5')
        with h5py.File(category_file_path, 'w') as h5file:
            h5file.create_dataset('main', data=np.array(predictions))



