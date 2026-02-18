import os
import random
import argparse
import xml.etree.ElementTree as ET
import json  # ВАЖНО: Добавлен импорт JSON
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from faker import Faker
from augmentor import ImageAugmentor

# Настройка Faker
fake = Faker('ru_RU')

male_patronymics = ['Иванович', 'Петрович', 'Сергеевич', 'Александрович', 'Михайлович', 'Дмитриевич']
female_patronymics = ['Ивановна', 'Петровна', 'Сергеевна', 'Александровна', 'Михайловна', 'Дмитриевна']


def parse_cvat_xml(xml_path):
    """
    Парсит XML от CVAT в формате Image 1.1 (теги <image> и <box>).
    """
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"Файл разметки не найден: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    boxes = {}

    for image in root.findall("image"):
        for box in image.findall("box"):
            label_name = box.get("label")

            xtl = float(box.get("xtl"))
            ytl = float(box.get("ytl"))
            xbr = float(box.get("xbr"))
            ybr = float(box.get("ybr"))
            rotation = float(box.get("rotation", "0"))

            width = xbr - xtl
            height = ybr - ytl
            cx = xtl + width / 2
            cy = ytl + height / 2

            boxes.setdefault(label_name, []).append({
                "label": label_name,
                "xtl": xtl, "ytl": ytl,
                "xbr": xbr, "ybr": ybr,
                "w": width, "h": height,
                "cx": cx, "cy": cy,
                "rotation": rotation
            })

    print(f"📦 Загружена разметка для полей: {list(boxes.keys())}")
    return boxes


def generate_data():
    """Генерирует случайные данные для одного паспорта"""
    is_male = random.choice([True, False])
    surname = fake.last_name_male() if is_male else fake.last_name_female()
    name = fake.first_name_male() if is_male else fake.first_name_female()
    patronymic = random.choice(male_patronymics if is_male else female_patronymics)

    return {
        'surname': surname,
        'name': name,
        'patronymic': patronymic,
        'issued_by': f"ОУФМС РОССИИ ПО {random.choice(['ГОР. МОСКВЕ', 'МОСКОВСКОЙ ОБЛ.'])} В {random.choice(['ЦАО', 'ЗАО', 'СВАО'])}",
        'issue_date': fake.date_between('-10y', '-1y').strftime('%d.%m.%Y'),
        'department_code': f"{random.randint(100, 999):03d}-{random.randint(100, 999):03d}",
        'passport_series': f"{random.randint(10, 99):02d} {random.randint(10, 99):02d}",
        'passport_number': f"{random.randint(100000, 999999):06d}",
        'sex': 'МУЖ.' if is_male else 'ЖЕН.',
        'birth_date': fake.date_of_birth(minimum_age=14, maximum_age=60).strftime('%d.%m.%Y'),
        'birth_place': f"ГОР. {fake.city().upper()}"
    }


_cached_font_path = None


def find_font():
    """Находит подходящий шрифт TrueType в системе и кэширует путь."""
    global _cached_font_path
    if _cached_font_path and os.path.exists(_cached_font_path):
        return _cached_font_path

    font_paths = [
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf"
    ]

    for path in font_paths:
        if os.path.exists(path):
            _cached_font_path = path
            print(f"✅ Найден и используется шрифт: {path}")
            return path

    raise RuntimeError("Не удалось найти подходящий шрифт TrueType.")


def get_font_for_box(box, is_vertical=False, max_font_size=42):
    """Подбирает размер шрифта под высоту/ширину бокса"""
    box_limit = box['w'] if is_vertical else box['h']
    target_size = int(box_limit * 0.65)
    target_size = min(target_size, max_font_size)
    target_size = max(target_size, 10)
    font_path = find_font()
    return ImageFont.truetype(font_path, target_size)


def draw_rotated_text(img, box, text, color=(0, 0, 0)):
    """Рисует текст с учетом вращения и специфики паспорта"""
    is_vertical_field = 'passport' in box['label'].lower()
    font = get_font_for_box(box, is_vertical=is_vertical_field)

    temp_dim = int(max(box['w'], box['h']) * 2.5)
    txt_layer = Image.new('RGBA', (temp_dim, temp_dim), (0, 0, 0, 0))
    draw = ImageDraw.Draw(txt_layer)

    try:
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
    except TypeError:
        text_w, text_h = draw.textsize(text, font=font)

    draw.text(((temp_dim - text_w) / 2, (temp_dim - text_h) / 2), text, font=font, fill=color + (255,))

    pil_rotation_angle = -box['rotation']
    if is_vertical_field:
        pil_rotation_angle -= 90

    rotated_txt = txt_layer.rotate(pil_rotation_angle, resample=Image.BICUBIC, expand=True)
    paste_x = int(box['cx'] - rotated_txt.width / 2)
    paste_y = int(box['cy'] - rotated_txt.height / 2)
    img.paste(rotated_txt, (paste_x, paste_y), rotated_txt)


def fill_template(template_path, boxes, output_dir, file_prefix, count_idx, augmentor, apply_aug_prob):
    """Создает одно изображение паспорта и JSON разметку"""
    data = generate_data()

    try:
        img = Image.open(template_path).convert('RGBA')
    except FileNotFoundError:
        print(f"❌ Ошибка: Шаблон {template_path} не найден!")
        return

    text_color = (35, 30, 30)
    red_color = (35, 30, 30)

    for label_name, bboxes in boxes.items():
        if label_name in data:
            value = str(data[label_name])
            color = red_color if 'passport' in label_name else text_color
            for box in bboxes:
                draw_rotated_text(img, box, value, color)

    img = img.convert('RGB')

    if random.random() < apply_aug_prob:
        img = augmentor.process(img)
        print(f"    ✨ Аугментация применена.")

    # --- ЛОГИКА СОХРАНЕНИЯ (ОБНОВЛЕНА) ---
    timestamp = int(datetime.now().timestamp())
    # Соль нужна для избежания конфликта имен при мультипроцессорном запуске
    salt = random.randint(10000, 99999)
    base_filename = f"{file_prefix}_{timestamp}_{count_idx + 1}_{salt}"

    # Пути
    image_path = os.path.join(output_dir, f"{base_filename}.png")
    json_path = os.path.join(output_dir, f"{base_filename}.json")

    # Сохраняем картинку
    img.save(image_path, quality=95)

    # Сохраняем JSON (Ground Truth)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"✅ [{count_idx + 1}] Сохранено: {base_filename}.png и .json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Генератор синтетических паспортов")
    parser.add_argument('--count', type=int, default=5, help='Количество генерируемых изображений')
    parser.add_argument('--template', type=str, default='Sloi-1.jpg', help='Путь к шаблону')
    parser.add_argument('--xml', type=str, default='annotations.xml', help='Путь к файлу разметки CVAT')
    parser.add_argument('--out', type=str, default='generated', help='Папка для сохранения')
    parser.add_argument('--aug-prob', type=float, default=1 / 3, help='Вероятность применения аугментаций')
    parser.add_argument('--aug-internal-prob', type=float, default=0.7, help='Вероятность применения каждого искажения')

    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    try:
        augmentor = ImageAugmentor(probability=args.aug_internal_prob)
        boxes_data = parse_cvat_xml(args.xml)

        if not boxes_data:
            print("⚠️ Внимание: В XML файле не найдено ни одного бокса.")
        else:
            find_font()
            print(f"🚀 Начинаем генерацию {args.count} шт...")
            for i in range(args.count):
                fill_template(args.template, boxes_data, args.out, "passport", i, augmentor, args.aug_prob)
            print("🎉 Генерация завершена!")

    except Exception as e:
        print(f"❌ Произошла критическая ошибка: {e}")