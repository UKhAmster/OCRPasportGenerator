import argparse
import torch
import re
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel


def recognize_document(image_path, doc_type):
    # Определяем пути и промпты в зависимости от типа документа
    if doc_type == "passport":
        model_path = "models_ready/donut_passport_v1"
        task_prompt = "<s_passport>"
    elif doc_type == "registration":
        model_path = "models_ready/donut_registration_v1"
        task_prompt = "<s_registration>"
    else:
        raise ValueError("Тип документа должен быть 'passport' или 'registration'")

    print(f"⏳ Загрузка модели из {model_path}...")
    try:
        processor = DonutProcessor.from_pretrained(model_path)
        model = VisionEncoderDecoderModel.from_pretrained(model_path)
    except Exception as e:
        print(f"❌ Ошибка загрузки модели (возможно, указан неверный путь): {e}")
        return

    # Переносим модель на видеокарту
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚡ Используемое устройство: {device.upper()}")
    model.to(device)
    model.eval()

    print(f"🖼️ Обработка изображения: {image_path}")
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"❌ Ошибка открытия картинки: {e}")
        return

    # Подготавливаем картинку для модели
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

    # Задаем стартовый токен (наш промпт)
    decoder_input_ids = processor.tokenizer(
        task_prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)

    print("🧠 Нейросеть читает документ...")
    # Запускаем генерацию
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

    # Декодируем токены обратно в читаемый текст
    sequence = processor.batch_decode(outputs.sequences)[0]

    # Очищаем технические токены (pad, eos и сам стартовый промпт)
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = re.sub(r"^" + re.escape(task_prompt), "", sequence).strip()

    print("\n" + "=" * 50)
    print("✅ РЕЗУЛЬТАТ РАСПОЗНАВАНИЯ:")
    print("=" * 50)
    print(sequence)
    print("=" * 50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Тестирование OCR моделей (Паспорт / Прописка)")
    parser.add_argument('--image', type=str, required=True, help='Путь к тестовой картинке')
    parser.add_argument('--type', type=str, choices=['passport', 'registration'], required=True, help='Тип документа')

    args = parser.parse_args()
    recognize_document(args.image, args.type)