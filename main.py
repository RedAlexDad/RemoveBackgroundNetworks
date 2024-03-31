import os

from PIL import Image
from tqdm import tqdm
import xml.etree.ElementTree as ET

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import torchvision
import torchvision.transforms as transforms


class VOCDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.image_dir = os.path.join(root_dir, 'JPEGImages')
        self.mask_dir = os.path.join(root_dir, 'SegmentationObject')
        self.class_dir = os.path.join(root_dir, 'SegmentationClass')

        self.annotation_dir = os.path.join(root_dir, 'Annotations')
        self.annotations = self.load_annotations()

        # Создаем список пар изображений и их масок
        self.dataset = self.find_valid_files()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name, mask_name = self.dataset[idx]

        # Проверка наличия маски
        if not os.path.exists(mask_name):
            # Маска отсутствует, пропускаем это изображение
            print(f"Warning: No mask found for image {img_name}")
            # Возвращаем пустые тензоры вместо изображения и маски
            return torch.empty(3, 256, 256), torch.empty(256, 256)

        image = Image.open(img_name).convert('RGB')
        mask = Image.open(mask_name)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return [image, mask]  # Возвращаем тензоры image и mask

    def find_valid_files(self):
        dataset = []
        count_exist = 0
        count_no_exist = 0
        for filename in os.listdir(self.image_dir):
            if filename.endswith('.jpg'):
                image_name = filename
                mask_name = filename.replace('.jpg', '.png')
                mask_path = os.path.join(self.mask_dir, mask_name)
                if os.path.exists(mask_path):
                    dataset.append((os.path.join(self.image_dir, image_name), mask_path))
                    count_exist += 1
                else:
                    count_no_exist += 1
        print(f'Количество отсутствующих файлов {count_no_exist}')
        print(f'Количество присутствующих файлов {count_exist}')
        print(f'Процент от всего: {count_exist / (count_exist + count_no_exist) * 100}%')
        return dataset

    def load_annotations(self):
        annotations = {}
        for filename in os.listdir(self.annotation_dir):
            if filename.endswith('.xml'):
                annotation_path = os.path.join(self.annotation_dir, filename)
                image_id = os.path.splitext(filename)[0]
                annotations[image_id] = self.parse_annotation(annotation_path)
        return annotations

    def parse_annotation(self, annotation_path):
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        objects = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            bbox = obj.find('bndbox')
            xmin = int(float(bbox.find('xmin').text))
            ymin = int(float(bbox.find('ymin').text))
            xmax = int(float(bbox.find('xmax').text))
            ymax = int(float(bbox.find('ymax').text))
            objects.append({'name': name, 'bbox': (xmin, ymin, xmax, ymax)})
        return objects

    def get_annotation(self, image_id):
        return self.annotations.get(image_id, [])

    def split_dataset(self, train_test_split=0.8):
        # Вычисляем размер обучающей выборки
        train_size = int(train_test_split * len(self.dataset))
        # Размер тестовой выборки
        test_size = len(self.dataset) - train_size

        print(f"Размер обучающей выборки: {train_size}")
        print(f"Размер тестовой выборки: {test_size}")

        # Разделение датасета на обучающую и тестовую выборки
        train_dataset, test_dataset = random_split(self.dataset, [train_size, test_size])

        return train_dataset, test_dataset

    def get_train_loader(self, train_dataset, batch_size=8, shuffle=True):
        return DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    def get_test_loader(self, test_dataset, batch_size=8, shuffle=False):
        return DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    # Пример преобразования данных для нейронной сети
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # Создание экземпляра класса датасета
    dataset = VOCDataset(root_dir='./VOCdevkit/VOC2012', transform=transform)

    # Разделение датасета на обучающую и тестовую выборки
    train_dataset, test_dataset = dataset.split_dataset(train_test_split=0.8)

    # Создание DataLoader для обучающей и тестовой выборок
    train_loader = dataset.get_train_loader(train_dataset, batch_size=8, shuffle=True)
    test_loader = dataset.get_test_loader(test_dataset, batch_size=8, shuffle=False)

    # Загрузка предварительно обученной модели DeepLabv3
    model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)

    # Определение функции потерь и оптимизатора
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Обучение модели
    model.train()
    for epoch in range(10):
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/10'):
            images, masks = batch  # Извлекаем изображения и маски из батча

            # Преобразование списка путей к файлам в список тензоров
            images = [transform(Image.open(img_name).convert('RGB')) for img_name in images]
            masks = [transform(Image.open(mask_name)) for mask_name in masks]

            # Объединение изображений и масок по оси пакета
            images = torch.stack(images)
            masks = torch.stack(masks)
            # Это изменение преобразует ваши 4-мерные one-hot маски в 3-мерные тензоры, где каждое значение представляет собой индекс класса.
            masks = torch.argmax(masks, dim=1)

            optimizer.zero_grad()
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            # print(f'Epoch {epoch + 1}/10, Loss: {loss.item()}')

    # Тестирование модели
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            images, masks = batch  # Извлекаем изображения и маски из батча

            # Преобразование изображений и масок в тензоры
            images = torch.stack(images)
            masks = torch.stack(masks)

            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            print(f'Test Loss: {loss.item()}')