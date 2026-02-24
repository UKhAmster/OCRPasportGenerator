import os
import json
import torch
import argparse
import re
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
from jiwer import cer


def evaluate(model_path, dataset_path, task_prompt):
    print(f"⏳ Загрузка модели из {model_path}...")
    processor = DonutProcessor.from_pretrained(model_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚡ Устройство: {device}")
    model.to(device)
    model.eval()

    metadata_file = os.path.join(dataset_path, "metadata.jsonl")
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Файл {metadata_file} не найден!")

    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = [json.loads(line) for line in f]

    total_cer = 0.0
    exact_matches = 0
    total_images = len(metadata)

    print(f"🚀 Начинаем валидацию {total_images} изображений...")

    for idx, item in enumerate(metadata):
        image_path = os.path.join(dataset_path, item["file_name"])

        # Достаем идеальный словарь из твоей метадаты
        ground_truth_dict = json.loads(item["ground_truth"])["gt_parse"]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"❌ Ошибка загрузки {image_path}: {e}")
            continue

        pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
        decoder_input_ids = processor.tokenizer(
            task_prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(device)

        with torch.no_grad():
            outputs = model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=768,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
                bad_words_ids=[[processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )

        # 1. Получаем сырую строку с тегами
        sequence = processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
        sequence = re.sub(r"^" + re.escape(task_prompt), "", sequence).strip()

        # 2. Магия: превращаем теги модели обратно в словарь!
        predicted_dict = processor.token2json(sequence)

        # 3. Считаем метрики (сравниваем строковые репрезентации словарей для CER)
        truth_str = json.dumps(ground_truth_dict, sort_keys=True, ensure_ascii=False)
        pred_str = json.dumps(predicted_dict, sort_keys=True, ensure_ascii=False)

        current_cer = cer(truth_str, pred_str)
        total_cer += current_cer

        if ground_truth_dict == predicted_dict:
            exact_matches += 1
            status = "✅ ИДЕАЛЬНО"
        else:
            status = f"❌ ОШИБКА (CER: {current_cer:.2f})"

        print(f"[{idx + 1}/{total_images}] {item['file_name']} | {status}")
        # Если интересно смотреть, где модель ошибается, раскомментируй строки ниже:
        # if ground_truth_dict != predicted_dict:
        #     print(f"   Ожидалось: {truth_str}")
        #     print(f"   Получено:  {pred_str}")

    # Финальные результаты
    avg_cer = total_cer / total_images
    accuracy = (exact_matches / total_images) * 100

    print("\n" + "=" * 50)
    print(f"📊 РЕЗУЛЬТАТЫ ВАЛИДАЦИИ ({dataset_path})")
    print("=" * 50)
    print(f"Всего изображений: {total_images}")
    print(f"Идеальных совпадений (Точность): {accuracy:.2f}%")
    print(f"Средняя ошибка по символам (CER): {avg_cer:.4f} (ближе к 0 = лучше)")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Массовая валидация моделей")
    parser.add_argument('--type', type=str, choices=['passport', 'registration'], required=True)
    args = parser.parse_args()

    if args.type == "passport":
        evaluate("models_ready/donut_passport_v1", "dataset/val_passport", "<s_passport>")
    elif args.type == "registration":
        evaluate("models_ready/donut_registration_v1", "dataset/val_registration", "<s_registration>")