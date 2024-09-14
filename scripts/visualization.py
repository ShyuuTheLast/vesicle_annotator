import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import find_contours

# Function to visualize image, prediction, and mask
def generate_html(image_paths, save_dir):
    # Sort images by prediction type
    dvh_images = []  # DVH should be checked first
    dv_images = []
    cv_images = []

    for image_path in image_paths:
        if 'DVH' in image_path:
            dvh_images.append(image_path)
        elif 'DV' in image_path:
            dv_images.append(image_path)
        elif 'CV' in image_path:
            cv_images.append(image_path)

    # Generate HTML content
    html_content = '<html><body>\n<h1>Model Predictions</h1>\n'

    # Add sections for each prediction type
    if dvh_images:
        html_content += '<h2>DVH Predictions</h2>\n'
        for image_path in dvh_images:
            relative_path = os.path.relpath(image_path, save_dir)
            html_content += f'<img src="{relative_path}" alt="DVH Prediction Image" style="width:100%;">\n'

    if dv_images:
        html_content += '<h2>DV Predictions</h2>\n'
        for image_path in dv_images:
            relative_path = os.path.relpath(image_path, save_dir)
            html_content += f'<img src="{relative_path}" alt="DV Prediction Image" style="width:100%;">\n'

    if cv_images:
        html_content += '<h2>CV Predictions</h2>\n'
        for image_path in cv_images:
            relative_path = os.path.relpath(image_path, save_dir)
            html_content += f'<img src="{relative_path}" alt="CV Prediction Image" style="width:100%;">\n'

    html_content += '</body></html>'

    # Save the HTML file
    html_path = os.path.join(save_dir, 'index.html')
    with open(html_path, 'w') as html_file:
        html_file.write(html_content)
    print(f"Saved HTML file to {html_path}")

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

def visualize_image_with_prediction(image, mask, label, pred_label, save_path):
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
        plt.suptitle(f'True Label: {true_label_text}, Pred Label: {pred_label_text}', fontsize=18)
    else:
        plt.suptitle(f'Pred Label: {pred_label_text}', fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the figure
    save_path = os.path.join(save_path)
    plt.savefig(save_path)
    plt.close(fig)

def convert_to_one_hot(prediction, shape, num_classes):
    if np.any(prediction >= num_classes) or np.any(prediction < 0):
        raise ValueError("Prediction array contains values outside the valid range of classes.")

    one_hot_mask = np.zeros((num_classes, *shape), dtype=np.int32)
    for i in range(num_classes):
        one_hot_mask[i, ...] = (prediction == i)

    return one_hot_mask

def html_visualize(model, data_loader, save_dir, n_classes):
    """
    Evaluates the model on the provided data_loader, generates predictions, and saves visualizations.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        data_loader (torch.utils.data.DataLoader): DataLoader providing the test data.
        save_dir (str): Directory where the visualizations and HTML will be saved.
        n_classes (int): The number of classes in the classification problem.
    """
    model.eval()
    num_images_saved = 0  # Counter to keep track of saved images
    image_paths = []  # List to store paths of saved images
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for inputs, masks, label in data_loader:  # Adjusted to match the correct return order
            print(f"Processing batch of {inputs.size(0)} images.")
            inputs = inputs.float().unsqueeze(1)  # Adding channel dimension for grayscale
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            for i in range(inputs.size(0)):
                num_images_saved += 1  # Increment the counter

                image = inputs[i, 0].cpu().numpy()
                prediction = predicted[i].cpu().numpy()
                mask = masks[i].cpu().numpy()
                pred_label = predicted[i].cpu().item()

                true_label = label[i]

                # Convert the predicted labels to one-hot encoded mask
                prediction_one_hot = convert_to_one_hot(prediction, image.shape, n_classes)

                # Generate file name using label text for correct categorization
                pred_label_text = get_label_text(pred_label)
                save_path = os.path.join(save_dir, f'prediction_{num_images_saved}_{pred_label_text}.png')

                # Visualize the prediction and save the image
                visualize_image_with_prediction(image, mask, true_label, pred_label, save_path)
                image_paths.append(save_path)

                if num_images_saved % 64 == 0:
                    print(f"Checkpoint: {num_images_saved} images processed")

    # Generate HTML file to visualize all images
    generate_html(image_paths, save_dir)



