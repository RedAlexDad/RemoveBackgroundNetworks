import os

import numpy as np
from PIL import Image
from tqdm import tqdm
import xml.etree.ElementTree as ET

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import torchvision
import torchvision.transforms as transforms


class RemoveBackgroundNetworks(Dataset):
    def __init__(self, root_dir, transform=None, device='cpu'):
        self.device = device
        self.root_dir = root_dir
        self.transform = transform

        self.image_dir = os.path.join(root_dir, 'JPEGImages')
        self.mask_dir = os.path.join(root_dir, 'SegmentationObject')
        self.class_dir = os.path.join(root_dir, 'SegmentationClass')

        self.annotation_dir = os.path.join(root_dir, 'Annotations')
        self.annotations = self.load_annotations()

        # Создаем список пар изображений и их масок
        self.dataset = self.find_valid_files()

        # Резульаты обучения
        self.train_losses = []
        self.test_losses = []

    def set_device(self, device='cpu'):
        print(f'Есть ли CUDA? : {torch.cuda.is_available()}')
        if device == 'cpu':
            self.device = 'cpu'
        else:
            self.device = 'cuda'

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

        # Преобразование one-hot маски в тензор с индексами классов
        mask = torch.argmax(mask, dim=1)

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
        # print(f'Количество отсутствующих файлов {count_no_exist}')
        # print(f'Количество присутствующих файлов {count_exist}')
        # print(f'Процент от всего: {count_exist / (count_exist + count_no_exist) * 100}%')
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

    def train_model(self, model, criterion, optimizer, train_loader, num_epochs=10):
        model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
                # Извлекаем изображения и маски из батча
                images, masks = batch

                # Преобразование списка путей к файлам в список тензоров
                images = [transform(Image.open(img_name).convert('RGB')).to(self.device) for img_name in images]
                masks = [transform(Image.open(mask_name)).to(self.device) for mask_name in masks]

                images = torch.stack(images)
                masks = torch.stack(masks)
                # Это изменение преобразует ваши 4-мерные one-hot маски в 3-мерные тензоры, где каждое значение представляет собой индекс класса.
                masks = torch.argmax(masks, dim=1)

                optimizer.zero_grad()
                outputs = model(images)['out']
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                # print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

            # Сохранение значения функции потерь на каждой эпохе
            self.train_losses.append(epoch_loss / len(train_loader))
            print(f'\nEpoch {epoch + 1}/{num_epochs}, Loss: {self.train_losses[-1]}')

        # Сохранение модели
        torch.save(model.state_dict(), 'trained_model.pth')

    def evaluate_model(self, model, criterion, test_loader):
        model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Testing'):
                # Извлекаем изображения и маски из батча
                images, masks = batch

                # Преобразование списка путей к файлам в список тензоров
                images = torch.stack(
                    [transform(Image.open(img_name).convert('RGB')).to(self.device) for img_name in images])
                masks = torch.stack([transform(Image.open(mask_name)).to(self.device) for mask_name in masks])

                # Это изменение преобразует ваши 4-мерные one-hot маски в 3-мерные тензоры, где каждое значение представляет собой индекс класса.
                masks = torch.argmax(masks, dim=1)
                outputs = model(images)['out']
                loss = criterion(outputs, masks)
                self.test_losses.append(loss.item())
                # print(f'Test Loss: {loss.item()}')

    def set_model(self, model_path=None):
        model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False)
        state_dict = torch.load(model_path)

        # Удаление ненужных ключей из state_dict
        keys_to_remove = [key for key in state_dict.keys() if 'aux_classifier' in key]
        for key in keys_to_remove:
            del state_dict[key]

        # Загрузка оставшихся ключей
        model.load_state_dict(state_dict)

        return model

    def print_losses(self):
        if len(self.train_losses) != 0:
            print("Train Losses:")
            for epoch, loss in enumerate(self.train_losses):
                print(f"Epoch {epoch + 1}: {loss}")

        if len(self.test_losses) != 0:
            print("\nTest Losses:")
            for epoch, loss in enumerate(self.test_losses):
                print(f"Epoch {epoch + 1}: {loss}")

    def remove_background(self, image, mask):
        # Получение размеров изображения
        image_size = (image.shape[2], image.shape[1])  # Размеры (ширина, высота)

        # Преобразование тензора маски в изображение PIL
        mask_image = transforms.ToPILImage()(mask.byte())

        # Преобразование маски в одноканальное изображение
        mask_image = mask_image.convert('L')

        # Приведение размеров маски к размерам изображения
        mask_image = mask_image.resize(image_size)

        # Преобразование тензора изображения в изображение PIL
        image_pil = transforms.ToPILImage()(image.cpu().byte())

        # Применение маски к изображению
        color_background = (255, 255, 255)
        image_with_mask = Image.composite(image_pil, Image.new('RGB', image_size, color_background), mask_image)

        return image_with_mask


if __name__ == "__main__":
    # form_image = (128, 128)
    # form_image = (256, 256)
    form_image = (512, 512)

    # Пример преобразования данных для нейронной сети
    transform = transforms.Compose([
        transforms.Resize(form_image),
        transforms.ToTensor()
    ])

    # Создание экземпляра класса датасета
    RBN = RemoveBackgroundNetworks(root_dir='./VOCdevkit/VOC2012', transform=transform)
    # Ставим CUDA
    RBN.set_device(device='cpu')

    # Разделение датасета на обучающую и тестовую выборки
    train_dataset, test_dataset = RBN.split_dataset(train_test_split=0.8)

    # Создание DataLoader для обучающей и тестовой выборок
    train_loader = RBN.get_train_loader(train_dataset, batch_size=8, shuffle=True)
    test_loader = RBN.get_test_loader(test_dataset, batch_size=8, shuffle=False)

    # Загрузка предварительно обученной модели DeepLabv3
    model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
    model.to(RBN.device)

    # Определение функции потерь и оптимизатора
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if (os.path.exists('trained_model.pth')):
        model = RBN.set_model(model_path='trained_model.pth')  # Загрузка модели
        model.to(RBN.device)

        # RBN.evaluate_model(model=model, criterion=criterion, test_loader=test_loader)

        # Вывод результатов
        # RBN.print_losses()
    else:
        RBN.train_model(model=model, criterion=criterion, optimizer=optimizer, train_loader=train_loader, num_epochs=10)
        RBN.evaluate_model(model=model, criterion=criterion, test_loader=test_loader)

        # Вывод результатов
        RBN.print_losses()

    # Выключаем модель для предсказания
    # model.eval()
    # Использование метода для удаления фона из изображения
    # image_path = './VOCdevkit/VOC2012/JPEGImages/2008_001134.jpg'  # Путь к изображению
    # removed_background_image = RBN.remove_background(model=model, image_path=image_path)
    for i in range(10):
        image, mask = RBN[i]  # Получаем изображение и маску из вашего датасета
        result_image = RBN.remove_background(image, mask)
        result_image.show()  # Показываем результат

    # Вывод изображения с удаленным фоном
    # removed_background_image.show()