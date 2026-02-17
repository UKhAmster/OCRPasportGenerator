import os
import random
import argparse
import json
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont
from augmentor import ImageAugmentor


class PassportGenerator:
    def __init__(self, template_path, xml_path, fonts_dir, output_dir="generated"):
        self.template_path = template_path
        self.output_dir = output_dir

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.fonts = [os.path.join(fonts_dir, f) for f in os.listdir(fonts_dir)
                      if f.lower().endswith(('.ttf', '.otf'))]

        if not self.fonts:
            raise IOError(f"Не найдено шрифтов в папке: {fonts_dir}")

        self.fields = self._parse_cvat_xml(xml_path)

    def _parse_cvat_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        fields = {}
        image_node = root.find('image')

        def calc_metrics(xmin, ymin, xmax, ymax):
            height = ymax - ymin
            return {
                "x": xmin,
                "y_center": ymin + height / 2,
                "y_bottom": ymax,
                "h": height,
                "width": xmax - xmin
            }

        for box in image_node.findall('box'):
            fields[box.get('label')] = calc_metrics(
                float(box.get('xtl')), float(box.get('ytl')),
                float(box.get('xbr')), float(box.get('ybr'))
            )

        for poly in image_node.findall('polygon'):
            points = [tuple(map(float, p.split(','))) for p in poly.get('points').split(';')]
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            fields[poly.get('label')] = calc_metrics(min(xs), min(ys), max(xs), max(ys))

        return fields

    def _get_black_ink_color(self):
        base = random.randint(0, 30)
        return (
            base + random.randint(0, 10),
            base + random.randint(0, 10),
            base + random.randint(0, 20),
            random.randint(220, 255)
        )

    def generate_fake_data(self):
        data = {}
        streets = ["ЛЕНИНА", "МИРА", "САДОВАЯ", "ЖУКОВА", "ПУШКИНА"]
        cities = ["МОСКВА", "ХИМКИ", "ОДИНЦОВО", "ЛЮБЕРЦЫ"]

        if "Region" in self.fields: data["Region"] = "МОСКОВСКАЯ ОБЛ."
        if "District" in self.fields: data["District"] = "ОДИНЦОВСКИЙ Р-Н"
        if "city" in self.fields: data["city"] = f"ГОР. {random.choice(cities)}"
        if "street" in self.fields: data["street"] = f"УЛ. {random.choice(streets)}"
        if "house_number" in self.fields: data["house_number"] = str(random.randint(1, 199))
        if "korpus" in self.fields: data["korpus"] = random.choice(["1", "2", ""])
        if "stroenie" in self.fields: data["stroenie"] = random.choice(["1", "2", ""])
        if "apart_nmb" in self.fields: data["apart_nmb"] = str(random.randint(1, 150))
        return data

    def render(self, augmentor, apply_aug_prob, filename_prefix="handwritten"):
        img = Image.open(self.template_path).convert("RGBA")
        txt_layer = Image.new("RGBA", img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(txt_layer)

        data = self.generate_fake_data()

        # Проверка на наличие шрифтов
        if not self.fonts:
            raise IOError("Список шрифтов пуст!")

        font_path = random.choice(self.fonts)
        ink_color = self._get_black_ink_color()

        for label, coords in self.fields.items():
            if label not in data or not data[label]: continue
            text = data[label]

            # Логика размера шрифта
            size_multiplier = 1.2
            if label in ["house_number", "korpus", "stroenie", "apart_nmb"]:
                size_multiplier = 1.6

            # Безопасный расчет высоты шрифта
            font_size = int(coords['h'] * size_multiplier)
            if font_size <= 0: font_size = 12  # Защита от нулевого размера

            try:
                font = ImageFont.truetype(font_path, font_size)
            except Exception as e:
                print(f"Ошибка шрифта {font_path}: {e}")
                continue

            # Рандомизация позиции
            x = coords['x'] + random.randint(0, 10)
            # y_bottom - это низ бокса. Поднимаем текст на высоту шрифта + шум
            y = coords['y_bottom'] - font_size + random.randint(-5, 5)

            # Рисуем текст на прозрачном слое
            draw.text((x, y), text, font=font, fill=ink_color)

        # Слияние текста с шаблоном
        final_img = Image.alpha_composite(img, txt_layer).convert("RGB")  # Конвертируем в RGB для JPG

        # Применение аугментации с заданной вероятностью
        if random.random() < apply_aug_prob:
            final_img = augmentor.process(final_img)
            print(f"    ✨ Аугментация применена.")

        # --- СОХРАНЕНИЕ (ИЗМЕНЕНО) ---

        # Генерируем ID один раз, чтобы он совпал для обоих файлов
        file_id = random.randint(1000, 9999)
        base_filename = f"{filename_prefix}_{file_id}"

        # 1. Сохраняем картинку
        image_path = os.path.join(self.output_dir, f"{base_filename}.jpg")
        final_img.save(image_path, "JPEG", quality=random.randint(85, 98))

        # 2. Сохраняем JSON (Ground Truth)
        json_path = os.path.join(self.output_dir, f"{base_filename}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            # ensure_ascii=False позволяет сохранять кириллицу читаемой
            json.dump(data, f, ensure_ascii=False, indent=4)

        print(f"✅ Saved sample: {image_path} + JSON")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Генератор рукописных данных в паспорте.")
    parser.add_argument('--count', type=int, default=5, help='Количество изображений для генерации.')
    parser.add_argument('--template', type=str, default='img.png', help='Путь к файлу шаблона.')
    parser.add_argument('--xml', type=str, default='annotations1.xml', help='Путь к файлу разметки CVAT XML.')
    parser.add_argument('--fonts', type=str, default='fonts', help='Папка со шрифтами.')
    parser.add_argument('--out', type=str, default='generated', help='Папка для сохранения результатов.')
    parser.add_argument('--aug-prob', type=float, default=1/3, help='Вероятность применения всего набора аугментаций к изображению.')
    parser.add_argument('--aug-internal-prob', type=float, default=0.7, help='Вероятность применения каждого отдельного искажения внутри аугментатора.')
    args = parser.parse_args()

    try:
        augmentor = ImageAugmentor(probability=args.aug_internal_prob)
        gen = PassportGenerator(
            template_path=args.template,
            xml_path=args.xml,
            fonts_dir=args.fonts,
            output_dir=args.out
        )
        print(f"🚀 Начинаем генерацию {args.count} рукописных образцов...")
        for i in range(args.count):
            gen.render(augmentor, args.aug_prob, f"handwritten_{i}")
        print("🎉 Генерация завершена!")
    except Exception as e:
        print(f"❌ Произошла критическая ошибка: {e}")
