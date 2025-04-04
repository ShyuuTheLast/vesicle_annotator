import os
import re
import h5py
import numpy as np

def extract_labels_from_txt(txt_path):
    """
    Extract (vesicleID, labelIndex) pairs from a txt file.
    """
    with open(txt_path, 'r') as f:
        content = f.read().strip()
    matches = re.findall(r'\((\d+):(\d+)\)', content)
    return {int(vid): int(label) for vid, label in matches}

def merge_and_save_labels(label_dict, save_path):
    """
    Merge, sort by vesicle ID, and save as HDF5 file under 'main' dataset.
    """
    sorted_ids = sorted(label_dict.keys())
    sorted_labels = [label_dict[vid] for vid in sorted_ids]
    with h5py.File(save_path, 'w') as f:
        f.create_dataset('main', data=np.array(sorted_labels, dtype=np.int32))

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    html_results_dir = os.path.join(project_root, 'html results')
    data_dir = os.path.join(project_root, 'data')

    # TEMPORARY: Limit to these cell names for testing
    target_cells = {'KM4', 'KR5'}

    for folder_name in os.listdir(html_results_dir):
        if not folder_name.endswith(' txt'):
            continue

        cell_name = folder_name.replace(' txt', '')
        if cell_name not in target_cells:
            continue

        txt_folder_path = os.path.join(html_results_dir, folder_name)
        label_dict = {}

        for fname in os.listdir(txt_folder_path):
            if not fname.startswith('saved_') or not fname.endswith('.txt'):
                continue

            txt_path = os.path.join(txt_folder_path, fname)
            labels = extract_labels_from_txt(txt_path)
            label_dict.update(labels)

        if not label_dict:
            print(f"No valid labels found for {cell_name}. Skipping.")
            continue

        save_subdir = os.path.join(data_dir, cell_name)
        os.makedirs(save_subdir, exist_ok=True)
        save_path = os.path.join(save_subdir, 'label.h5')
        merge_and_save_labels(label_dict, save_path)
        print(f"Saved label.h5 for {cell_name} with {len(label_dict)} entries.")

def test_cell_shapes(cell_name):
    """
    Load im.h5, mask.h5, and label.h5 for the given cell and print shapes.
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    cell_dir = os.path.join(base_dir, 'data', cell_name)

    im_file = os.path.join(cell_dir, 'im.h5')
    mask_file = os.path.join(cell_dir, 'mask.h5')
    label_file = os.path.join(cell_dir, 'label.h5')

    with h5py.File(im_file, 'r') as f:
        im_data = f['main'][:]
        print("Image shape:", im_data.shape)

    with h5py.File(mask_file, 'r') as f:
        mask_data = f['main'][:]
        print("Mask shape:", mask_data.shape)

    with h5py.File(label_file, 'r') as f:
        label_data = f['main'][:]
        print("Label shape:", label_data.shape)

    print("Image count:", len(im_data))
    print("Mask count: ", len(mask_data))
    print("Label count:", len(label_data))

if __name__ == '__main__':
    main()

    test_result = True
    if test_result:
        cell = 'KM4'
        test_cell_shapes(cell)