import os
import random
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont, ImageFilter


class PassportGenerator:
    def __init__(self, template_path, xml_path, fonts_dir, output_dir="generated"):
        self.template_path = template_path
        self.output_dir = output_dir

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.fonts = [os.path.join(fonts_dir, f) for f in os.listdir(fonts_dir)
                      if f.lower().endswith(('.ttf', '.otf'))]

        if not self.fonts:
            print(f"[Warning] Нет шрифтов в {fonts_dir}!")

        self.fields = self._parse_cvat_xml(xml_path)

    def _parse_cvat_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        fields = {}
        image_node = root.find('image')

        def calc_metrics(xmin, ymin, xmax, ymax):
            height = ymax - ymin
            # Важно: запоминаем центр и нижнюю границу (baseline)
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
        # Темный, насыщенный цвет (имитация гелевой или шариковой ручки)
        base = random.randint(0, 30)
        return (
            base + random.randint(0, 10),
            base + random.randint(0, 10),
            base + random.randint(0, 20),
            random.randint(220, 255)  # Почти непрозрачный
        )

    def generate_fake_data(self):
        data = {}
        # Данные капсом, как в паспорте
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

    def render(self, filename_prefix="passport"):
        img = Image.open(self.template_path).convert("RGBA")
        txt_layer = Image.new("RGBA", img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(txt_layer)

        data = self.generate_fake_data()
        font_path = random.choice(self.fonts) if self.fonts else None
        ink_color = self._get_black_ink_color()

        for label, coords in self.fields.items():
            if label not in data or not data[label]: continue

            text = data[label]

            # --- ЛОГИКА РАЗМЕРА (КЛЮЧЕВОЕ ИЗМЕНЕНИЕ) ---
            # Базовый коэффициент увеличения относительно высоты бокса
            size_multiplier = 1.5

            # Для крупных полей (Регион, Город) делаем шрифт еще больше
            if label in ["Region", "city", "District", "street"]:
                size_multiplier = 1.2
            elif label in ["house_number", "korpus", "stroenie", "apart_nmb"]:
                size_multiplier = 1.6

            font_size = int(coords['h'] * size_multiplier)

            if font_path:
                font = ImageFont.truetype(font_path, font_size)
            else:
                font = ImageFont.load_default()

            # --- ЛОГИКА ПОЗИЦИИ ---
            # Смещаем текст, чтобы он визуально стоял на строке, а не висел в воздухе.
            # coords['y_bottom'] - это нижняя черта.
            # Поднимаем текст примерно на 80% от размера шрифта вверх от черты.
            x = coords['x'] + random.randint(0, 10)
            y = coords['y_bottom'] - (font_size * 0.75) + random.randint(-5, 5)

            draw.text((x, y), text, font=font, fill=ink_color)

        # Легкий блюр для слияния с бумагой
        txt_layer = txt_layer.filter(ImageFilter.GaussianBlur(radius=0.4))
        final_img = Image.alpha_composite(img, txt_layer)

        save_path = os.path.join(self.output_dir, f"{filename_prefix}_{random.randint(100, 999)}.jpg")
        final_img.convert("RGB").save(save_path, "JPEG", quality=95)
        print(f"Saved optimized sample: {save_path}")


# Запуск
if __name__ == "__main__":
    # Убедись, что пути верные
    gen = PassportGenerator(
        template_path="img.png",
        xml_path="annotations1.xml",
        fonts_dir="fonts"
    )
    for i in range(1):
        gen.render(f"big_text_{i}")