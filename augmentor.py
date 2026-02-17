import cv2
import numpy as np
import random
from PIL import Image, ImageEnhance


class ImageAugmentor:
    def __init__(self, probability=0.5):
        self.prob = probability

    def process(self, pil_image: Image.Image) -> Image.Image:
        # 1. Повороты на 90/180/270 градусов (Критично для сканера!)
        # Это применяем с вероятностью 70%, так как люди редко кладут идеально ровно
        if random.random() < 0.7:
            pil_image = self._apply_random_rotation_90(pil_image)

        # Конвертация в OpenCV для геометрии
        cv_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # 2. Легкий перекос (Skew) - бумага легла чуть криво
        if random.random() < self.prob:
            cv_img = self._apply_slight_skew(cv_img)

        # 3. Шум сканера (зернистость на высоких DPI)
        if random.random() < self.prob:
            cv_img = self._apply_scanner_noise(cv_img)

        # Обратно в PIL для работы с цветом
        pil_image = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

        # 4. Яркость и Контраст (Лампа сканера)
        if random.random() < self.prob:
            pil_image = self._apply_exposure_jitter(pil_image)

        # 5. Бинаризация (Имитация ч/б сканирования или режима "Документ")
        if random.random() < 0.1:  # Редко, но бывает
            pil_image = self._apply_binarization_look(pil_image)

        return pil_image

    def _apply_random_rotation_90(self, img):
        """Сканер может выдать картинку в любой ориентации"""
        angle = random.choice([0, 90, 180, 270])
        if angle == 0: return img
        return img.rotate(angle, expand=True)

    def _apply_slight_skew(self, img):
        """Поворот на +/- 1-3 градуса"""
        h, w = img.shape[:2]
        angle = random.uniform(-2.5, 2.5)  # Небольшой угол
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        # Заливаем края белым (сканер дает белый фон), а не черным
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    def _apply_scanner_noise(self, img):
        """Легкий цифровой шум CCD-матрицы"""
        row, col, ch = img.shape
        mean = 0
        var = random.uniform(2, 10)  # Очень слабый шум
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        noisy = img + gauss
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def _apply_exposure_jitter(self, img):
        """Имитация разных настроек гаммы сканера"""
        # Яркость
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.8, 1.3))
        # Контраст (сканеры часто "пережаривают" контраст в режиме текста)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.8, 1.5))
        return img

    def _apply_binarization_look(self, img):
        """Имитация режима 'Текст' (удаление полутонов)"""
        # Переводим в оттенки серого и обратно, повышая контраст до предела
        gray = img.convert('L')
        # Порог (threshold)
        thresh = random.randint(100, 200)
        fn = lambda x: 255 if x > thresh else 0
        return gray.point(fn, mode='1').convert('RGB')