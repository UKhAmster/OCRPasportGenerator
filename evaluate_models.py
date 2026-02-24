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
        ground_truth = item["ground_truth"]

        image = Image.open(image_path).convert("RGB")
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

        # Декодируем и очищаем прогноз
        prediction = processor.batch_decode(outputs.sequences)[0]
        prediction = prediction.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
        prediction = re.sub(r"^" + re.escape(task_prompt), "", prediction).strip()

        # Считаем метрики
        current_cer = cer(ground_truth, prediction)
        total_cer += current_cer

        if ground_truth == prediction:
            exact_matches += 1

        print(f"[{idx+1}/{total_images}] Обработан: {item['file_name']} | CER: {current_cer:.4f}")

    # Финальные результаты
    avg_cer = total_cer / total_images
    accuracy = (exact_matches / total_images) * 100

    print("\n" + "="*50)
    print(f"📊 РЕЗУЛЬТАТЫ ВАЛИДАЦИИ ({dataset_path})")
    print("="*50)
    print(f"Всего изображений: {total_images}")
    print(f"Идеальных совпадений (Exact Match): {accuracy:.2f}%")
    print(f"Средняя ошибка по символам (Average CER): {avg_cer:.4f} (ближе к 0 = лучше)")
    print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Массовая валидация моделей")
    parser.add_argument('--type', type=str, choices=['passport', 'registration'], required=True)
    args = parser.parse_args()

    if args.type == "passport":
        evaluate("models_ready/donut_passport_v1", "dataset/val_passport", "<s_passport>")
    elif args.type == "registration":
        evaluate("models_ready/donut_registration_v1", "dataset/val_registration", "<s_registration>")