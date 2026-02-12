import cv2
import numpy as np
import random
from PIL import Image, ImageFilter


class ImageAugmentor:
    def __init__(self, probability=0.7):
        """
        :param probability: Вероятность применения каждой аугментации (0.0 - 1.0)
        """
        self.prob = probability

    def process(self, pil_image: Image.Image) -> Image.Image:
        """
        Главный метод: принимает PIL Image, возвращает PIL Image с искажениями.
        """
        # Конвертируем PIL -> OpenCV (BGR)
        cv_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # 1. Случайное размытие (Focus Blur)
        if random.random() < self.prob:
            cv_img = self._apply_gaussian_blur(cv_img)

        # 2. Смаз от движения (Motion Blur) - частая проблема фото с рук
        if random.random() < self.prob:
            cv_img = self._apply_motion_blur(cv_img)

        # 3. Шум матрицы (ISO Noise)
        if random.random() < self.prob:
            cv_img = self._apply_iso_noise(cv_img)

        # 4. Снижение качества (Downscale -> Upscale)
        if random.random() < self.prob:
            cv_img = self._apply_low_res(cv_img)

        # 5. JPEG артефакты
        if random.random() < self.prob:
            cv_img = self._apply_jpeg_compression(cv_img)

        # Конвертируем обратно OpenCV -> PIL
        return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

    def _apply_gaussian_blur(self, img):
        # Выбираем нечетное ядро: 3, 5, 7
        ksize = random.choice([3, 5, 7])
        return cv2.GaussianBlur(img, (ksize, ksize), 0)

    def _apply_motion_blur(self, img):
        # Имитация дрожания камеры
        ksize = random.choice([5, 7, 9, 12])
        # Генерируем ядро смаза
        kernel_motion_blur = np.zeros((ksize, ksize))
        # Случайный угол смаза
        if random.choice([True, False]):
            kernel_motion_blur[int((ksize - 1) / 2), :] = np.ones(ksize)  # Горизонтальный
        else:
            kernel_motion_blur[:, int((ksize - 1) / 2)] = np.ones(ksize)  # Вертикальный

        kernel_motion_blur /= ksize
        return cv2.filter2D(img, -1, kernel_motion_blur)

    def _apply_iso_noise(self, img):
        # Гауссовский шум (зернистость)
        row, col, ch = img.shape
        mean = 0
        var = random.uniform(10, 50)  # Сила шума
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = img + gauss
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def _apply_low_res(self, img):
        # Уменьшаем в 2-3 раза и возвращаем обратно (пикселизация)
        h, w = img.shape[:2]
        ratio = random.uniform(1.5, 3.0)
        small = cv2.resize(img, (int(w / ratio), int(h / ratio)), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    def _apply_jpeg_compression(self, img):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(50, 85)]
        result, encimg = cv2.imencode('.jpg', img, encode_param)
        decimg = cv2.imdecode(encimg, 1)
        return decimg