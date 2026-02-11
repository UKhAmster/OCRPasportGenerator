import os
import random
import argparse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from PIL import Image, ImageDraw, ImageFont
from faker import Faker
from num2words import num2words

# --- Configuration ---
fake = Faker('ru_RU')

male_patronymics = ['Иванович', 'Петрович', 'Сергеевич', 'Александрович', 'Михайлович', 'Дмитриевич']
female_patronymics = ['Ивановна', 'Петровна', 'Сергеевна', 'Александровна', 'Михайловна', 'Дмитриевна']

# Словарь для перевода месяцев
MONTHS_RU_GENITIVE = {
    1: 'января', 2: 'февраля', 3: 'марта', 4: 'апреля', 5: 'мая', 6: 'июня',
    7: 'июля', 8: 'августа', 9: 'сентября', 10: 'октября', 11: 'ноября', 12: 'декабря'
}

# --- Core Logic ---

def parse_cvat_polygon_xml(xml_path):
    """
    Парсит XML от CVAT в формате Image 1.1, используя <polygon> теги.
    Вычисляет минимальный ограничивающий прямоугольник (bounding box) для полигона.
    """
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"Файл разметки не найден: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = {}

    for image in root.findall("image"):
        for polygon in image.findall("polygon"):
            label_name = polygon.get("label").replace('&', '&amp;') # Нормализация имен
            points_str = polygon.get("points").split(';')
            points = [tuple(map(float, p.split(','))) for p in points_str]

            # Вычисляем bounding box из полигона
            xtl = min(p[0] for p in points)
            ytl = min(p[1] for p in points)
            xbr = max(p[0] for p in points)
            ybr = max(p[1] for p in points)

            # CVAT rotation - пока нет в polygon, но можно добавить, если нужно
            rotation = float(polygon.get("rotation", "0"))

            boxes.setdefault(label_name, []).append({
                "label": label_name,
                "xtl": xtl, "ytl": ytl, "xbr": xbr, "ybr": ybr,
                "w": xbr - xtl, "h": ybr - ytl,
                "cx": (xtl + xbr) / 2, "cy": (ytl + ybr) / 2,
                "rotation": rotation
            })

    print(f"📦 Загружена разметка для полей: {list(boxes.keys())}")
    return boxes


def generate_birth_certificate_data():
    """Генерирует данные для свидетельства о рождении с учетом исправленной логики."""
    is_male = random.choice([True, False])
    child_surname = fake.last_name_male() if is_male else fake.last_name_female()
    child_name = fake.first_name_male() if is_male else fake.first_name_female()
    child_patronymic = random.choice(male_patronymics if is_male else female_patronymics)

    birth_date = fake.date_of_birth(minimum_age=1, maximum_age=18)
    registration_date = birth_date + timedelta(days=random.randint(3, 30))
    issuance_date = registration_date + timedelta(days=random.randint(0, 5))

    city = fake.city()
    region = fake.region()
    country = "Российская Федерация"

    father_surname = fake.last_name_male()
    father_name = fake.first_name_male()
    father_patronymic = random.choice(male_patronymics)

    mother_surname = fake.last_name_female()
    mother_name = fake.first_name_female()
    mother_patronymic = random.choice(female_patronymics)


    return {
        # ИСПРАВЛЕННАЯ ЛОГИКА
        'FirstName': child_surname,
        'Surname&amp;patronymic': f"{child_name} {child_patronymic}",
        'FathersFirstname': father_surname,
        'FathersSurname': f"{father_name} {father_patronymic}",
        'MothersFirstname': mother_surname,
        'MothersSurname&amp;patronymic': f"{mother_name} {mother_patronymic}",

        # Остальные поля
        'birthDate': birth_date.strftime('%d.%m.%Y'),
        'birthdate(bylettersDay&amp;Month)': f"{num2words(birth_date.day, lang='ru', to='ordinal')} {MONTHS_RU_GENITIVE[birth_date.month]}",
        'birthdate(bylettersyear)': f"{num2words(birth_date.year, lang='ru', to='year')} года",
        'birthPlace': city,
        'birthPlace(region)': region,
        'birthplace(country)': country,
        'dayofregistration': str(registration_date.day),
        'monthofregistration': MONTHS_RU_GENITIVE[registration_date.month],
        'yearofregistration': str(registration_date.year),
        'numberofcertificate': f"{random.randint(100, 999):03d}",
        'cityzenship': "Гражданство РФ",
        'nationality': "русский" if is_male else "русская",
        'mothersnationality': "русская",
        'placeofGovRegistration': f"Отдел ЗАГС {fake.city_name()} района, {fake.region()}",
        'dateOfissuance': str(issuance_date.day),
        'monthOfissuence': MONTHS_RU_GENITIVE[issuance_date.month],
        'yearOfissuence': str(issuance_date.year),
    }

# --- Font and Drawing Logic (copied from passport generator) ---

_cached_font_path = None

def find_font():
    global _cached_font_path
    if _cached_font_path and os.path.exists(_cached_font_path):
        return _cached_font_path
    font_paths = [
        "C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf"
    ]
    for path in font_paths:
        if os.path.exists(path):
            _cached_font_path = path
            print(f"✅ Найден и используется шрифт: {path}")
            return path
    raise RuntimeError("Не удалось найти подходящий шрифт TrueType.")

def get_font_for_box(box, max_font_size=32):
    box_limit = box['h']
    target_size = int(box_limit * 0.7)
    target_size = min(target_size, max_font_size)
    target_size = max(target_size, 10)
    font_path = find_font()
    return ImageFont.truetype(font_path, target_size)

def draw_rotated_text(img, box, text, color=(0, 0, 0)):
    font = get_font_for_box(box)
    temp_dim = int(max(box['w'], box['h']) * 2)
    txt_layer = Image.new('RGBA', (temp_dim, temp_dim), (0, 0, 0, 0))
    draw = ImageDraw.Draw(txt_layer)

    try:
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    except TypeError:
        text_w, text_h = draw.textsize(text, font=font)

    draw.text(((temp_dim - text_w) / 2, (temp_dim - text_h) / 2), text, font=font, fill=color + (255,))
    pil_rotation_angle = -box['rotation']
    rotated_txt = txt_layer.rotate(pil_rotation_angle, resample=Image.BICUBIC, expand=True)
    paste_x = int(box['cx'] - rotated_txt.width / 2)
    paste_y = int(box['cy'] - rotated_txt.height / 2)
    img.paste(rotated_txt, (paste_x, paste_y), rotated_txt)

# --- Main Execution ---

def fill_template(template_path, boxes, output_dir, file_prefix, count_idx):
    data = generate_birth_certificate_data()
    try:
        img = Image.open(template_path).convert('RGBA')
    except FileNotFoundError:
        print(f"❌ Ошибка: Шаблон {template_path} не найден!")
        return

    text_color = (10, 10, 10)
    for label_name, bboxes in boxes.items():
        if label_name in data:
            value = str(data[label_name])
            for box in bboxes:
                draw_rotated_text(img, box, value, text_color)

    filename = f"{file_prefix}_{int(datetime.now().timestamp())}_{count_idx + 1}.png"
    save_path = os.path.join(output_dir, filename)
    img.convert('RGB').save(save_path, quality=95)
    print(f"✅ [{count_idx + 1}] Сохранено: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Генератор свидетельств о рождении")
    parser.add_argument('--count', type=int, default=1, help='Количество изображений')
    parser.add_argument('--template', type=str, default='img_1.png', help='Путь к шаблону')
    parser.add_argument('--xml', type=str, default='annotations2.xml', help='Путь к CVAT XML')
    parser.add_argument('--out', type=str, default='generated', help='Папка для сохранения')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    try:
        boxes_data = parse_cvat_polygon_xml(args.xml)
        if not boxes_data:
            print("⚠️ В XML не найдено ни одного полигона.")
        else:
            find_font()
            print(f"🚀 Начинаем генерацию {args.count} шт...")
            for i in range(args.count):
                fill_template(args.template, boxes_data, args.out, "cert", i)
            print("🎉 Генерация завершена!")
    except Exception as e:
        print(f"❌ Произошла критическая ошибка: {e}")
