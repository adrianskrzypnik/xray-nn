from sklearn.model_selection import train_test_split
import os
import shutil

def split_data(source_dir, dest_dir, train_size=0.7, val_size=0.15, test_size=0.15):
    assert train_size + val_size + test_size == 1.0, "Podziały muszą sumować się do 1"

    classes = os.listdir(source_dir)
    for cls in classes:
        cls_path = os.path.join(source_dir, cls)
        images = os.listdir(cls_path)

        # Losowy podział
        train, temp = train_test_split(images, train_size=train_size, random_state=42)
        val, test = train_test_split(temp, test_size=test_size/(val_size + test_size), random_state=42)

        # Tworzenie katalogów docelowych
        for split, split_images in zip(['train', 'val', 'test'], [train, val, test]):
            split_dir = os.path.join(dest_dir, split, cls)
            os.makedirs(split_dir, exist_ok=True)
            for img in split_images:
                shutil.copy(os.path.join(cls_path, img), os.path.join(split_dir, img))


split_data('Chest X_Ray Dataset', 'dataset')
