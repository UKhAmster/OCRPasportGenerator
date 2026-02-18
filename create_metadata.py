import json
import os
from pathlib import Path


def create_metadata_jsonl(dataset_dir):
    print(f"🔍 Сканируем папку: {dataset_dir}")
    output_file = os.path.join(dataset_dir, "metadata.jsonl")

    valid_pairs = 0
    with open(output_file, 'w', encoding='utf-8') as out_f:
        # Ищем все JSON файлы в папке
        for json_path in Path(dataset_dir).rglob("*.json"):
            # Пропускаем сам файл metadata.jsonl, чтобы не зациклить чтение
            if json_path.name == "metadata.jsonl":
                continue

            # Ищем соответствующую картинку (сначала пробуем .png, затем .jpg)
            img_path = json_path.with_suffix(".png")
            if not img_path.exists():
                img_path = json_path.with_suffix(".jpg")

            if not img_path.exists():
                print(f"⚠️ Ошибка: Для {json_path.name} нет картинки. Пропускаем.")
                continue

            # 1. Читаем данные из нашего сгенерированного JSON
            with open(json_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)

            # 2. Формируем структуру, которую понимает трансформер Donut
            # Ключ 'gt_parse' обязателен для задачи парсинга (Document Parsing)
            donut_ground_truth = {
                "gt_parse": raw_data
            }

            # 3. Собираем финальную строку для JSONL
            # ВАЖНО: Значение по ключу 'ground_truth' должно быть СТРОКОЙ (поэтому делаем json.dumps дважды)
            jsonl_line = {
                "file_name": img_path.name,
                "ground_truth": json.dumps(donut_ground_truth, ensure_ascii=False)
            }

            # 4. Пишем в файл с новой строки
            out_f.write(json.dumps(jsonl_line, ensure_ascii=False) + "\n")
            valid_pairs += 1

    print(f"✅ Готово! Файл {output_file} создан. Обработано пар: {valid_pairs}")


if __name__ == "__main__":
    # Папки, в которых лежат сгенерированные файлы
    # Запускай это только после того, как сгенерируешь картинки!
    create_metadata_jsonl("dataset/train")

    # Если ты делал отдельную папку для валидации, раскомментируй строку ниже:
    # create_metadata_jsonl("dataset/validation")