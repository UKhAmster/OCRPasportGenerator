from faker import Faker
import random
from PIL import Image, ImageDraw, ImageFont
import os
import re
from datetime import datetime, timedelta
import math

male_patronymics = ['Иванович', 'Петрович', 'Сергеевич', 'Александрович', 'Михайлович']
female_patronymics = ['Ивановна', 'Петровна', 'Сергеевна', 'Александровна', 'Михайловна']

fake = Faker('ru_RU')


def parse_cvat_xml(xml_path):
    """Парсит bbox + rotation"""
    with open(xml_path, 'r', encoding='utf-8') as f:
        content = f.read()

    boxes = {}
    # С rotation
    pattern = r'<box\s+label="([^"]+)"[^>]*xtl="([^"]+)"\s+ytl="([^"]+)"\s+xbr="([^"]+)"\s+ybr="([^"]+)"[^>]*rotation="([^"]+)"'
    matches = re.findall(pattern, content, re.IGNORECASE)

    for label, xtl, ytl, xbr, ybr, rot in matches:
        boxes.setdefault(label, []).append({
            'xtl': float(xtl), 'ytl': float(ytl),
            'xbr': float(xbr), 'ybr': float(ybr),
            'rot': float(rot),
            'cx': (float(xtl) + float(xbr)) / 2,
            'cy': (float(ytl) + float(ybr)) / 2
        })

    print("📦 Boxes с поворотом:", list(boxes.keys()))
    return boxes


def generate_data():
    is_male = random.choice([True, False])
    surname = fake.last_name()
    name = fake.first_name_male() if is_male else fake.first_name_female()

    # ✅ ПРАВИЛЬНОЕ отчество
    patronymic = random.choice(male_patronymics if is_male else female_patronymics)

    return {
        'surname': surname,
        'name': name,
        'patronymic': patronymic,  # "Сергеевна", "Иванович"
        'issued_by': f"ОВД УВД по {random.choice(['Ленинскому району г.Москвы', 'Центральному району'])}",
        'issue_date': fake.date_between('-3y', '-1y').strftime('%d.%m.%Y'),
        'department_code': f"{random.randint(770, 779):03d}-{random.randint(40, 85):02d}",
        'passport_series': f"45{random.randint(10, 99):02d}",
        'passport_number': f"{random.randint(100, 599):03d} {random.randint(100, 999):03d}",
        'sex': 'М' if is_male else 'Ж',
        'birth_date': fake.date_of_birth(minimum_age=20, maximum_age=45).strftime('%d.%m.%Y'),
        'birth_place': f"{fake.city()} обл."
    }

def needs_rotation(label):
    """🔄 Поворот ТОЛЬКО для серии/номера"""
    return 'passport' in label.lower()


def draw_text_simple(draw, x, y, text, font):
    """Прямой текст"""
    draw.text((x + 1, y + 1), text, font=font, fill=(110, 110, 110, 255))
    draw.text((x, y), text, font=font, fill=(0, 0, 0, 255))


def draw_text_rotated(img, box, text, font):
    """✅ ПОВОРОТ 110° ВПРАВО (по часовой стрелке)"""

    txt_size = (300, 80)
    txt_img = Image.new('RGBA', txt_size, (0, 0, 0, 0))
    txt_draw = ImageDraw.Draw(txt_img)

    bbox = txt_draw.textbbox((0, 0), text, font=font)
    tx = (txt_size[0] - (bbox[2] - bbox[0])) // 2
    ty = (txt_size[1] - (bbox[3] - bbox[1])) // 2

    # Тень + текст
    txt_draw.text((tx + 1, ty + 1), text, font=font, fill=(100, 100, 100, 255))
    txt_draw.text((tx, ty), text, font=font, fill=(0, 0, 0, 255))

    # 🔄 ВПРАВО = ОТРИЦАТЕЛЬНЫЙ угол (-110°)
    rotated = txt_img.rotate(-90, expand=True, resample=Image.BICUBIC)

    paste_x = int(box['cx'] - rotated.width / 2)
    paste_y = int(box['cy'] - rotated.height / 2)

    img.paste(rotated, (paste_x, paste_y), rotated)
    print(f"🔄 '{text}' ВПРАВО -110° → ({paste_x},{paste_y})")
    return paste_x, paste_y


def fill_template(template_path, boxes, data):
    img = Image.open(template_path).convert('RGBA')
    draw = ImageDraw.Draw(img)  # 🔥 ГЛОБАЛЬНЫЙ draw

    # Фикс 14px
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 14)
    except:
        font = ImageFont.load_default()

    print(f"📐 {img.size}")

    for label, bboxes in boxes.items():
        if label in data:
            text = data[label]

            for i, box in enumerate(bboxes):
                xtl, ytl, xbr, ybr = int(box['xtl']), int(box['ytl']), int(box['xbr']), int(box['ybr'])

                # 📍 Левый край
                x = xtl + 2

                # 📏 Нижняя граница bbox
                text_bbox = draw.textbbox((0, 0), text, font=font)
                baseline_y = text_bbox[3]
                y = ybr - baseline_y

                if 'passport' in label.lower():
                    # 🔄 ПОВОРОТ серия/номер -110° ВПРАВО
                    txt_img = Image.new('RGBA', (300, 60), (0, 0, 0, 0))
                    txt_draw = ImageDraw.Draw(txt_img)
                    txt_draw.text((2, 1), text, font=font, fill=(110, 110, 110, 255))
                    txt_draw.text((0, 0), text, font=font, fill=(0, 0, 0, 255))
                    rotated = txt_img.rotate(-110, expand=True)

                    paste_x = int((xtl + xbr) / 2 - rotated.width / 2)
                    paste_y = int((ytl + ybr) / 2 - rotated.height / 2)

                    img.paste(rotated, (paste_x, paste_y), rotated)
                    print(f"🔄 {label}#{i + 1}: '{text}' -110° ({paste_x},{paste_y})")
                else:
                    # ➤ ПРЯМОЙ текст
                    draw.text((x + 1, y + 1), text, font=font, fill=(110, 110, 110, 255))
                    draw.text((x, y), text, font=font, fill=(0, 0, 0, 255))
                    print(f"➤ {label}: '{text}' ({x},{y}) ybr={ybr}")

    output = f'generated/FIXED_{int(datetime.now().timestamp())}.png'
    os.makedirs('generated', exist_ok=True)
    img.save(output, quality=98)
    print(f"✅ БЕЗ ОШИБОК: {output}")
    return output


if __name__ == "__main__":
    boxes = parse_cvat_xml('annotations.xml')
    data = generate_data()
    fill_template('Sloi-1.jpg', boxes, data)
