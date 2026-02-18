import os
import json
import argparse
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel, VisionEncoderDecoderConfig
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

# --- Базовые настройки ---
MODEL_REPO = "naver-clova-ix/donut-base"
MAX_LENGTH = 768
IMAGE_SIZE = (2560, 1920)  # Высокое разрешение, критично для мелкого шрифта и рукописи


class DonutDataset(Dataset):
    def __init__(self, dataset_path, processor):
        self.dataset_path = dataset_path
        self.processor = processor
        self.metadata = []

        # Загружаем метаданные
        metadata_file = os.path.join(dataset_path, "metadata.jsonl")
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Файл {metadata_file} не найден! Запусти create_metadata.py")

        with open(metadata_file, "r", encoding="utf-8") as f:
            for line in f:
                self.metadata.append(json.loads(line))

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        image_path = os.path.join(self.dataset_path, item["file_name"])
        image = Image.open(image_path).convert("RGB")

        # Подготовка картинки (перевод в тензоры)
        pixel_values = self.processor(image, return_tensors="pt").pixel_values

        # Подготовка текста (JSON -> токены)
        target_sequence = item["ground_truth"]
        input_ids = self.processor.tokenizer(
            target_sequence,
            add_special_tokens=False,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids

        labels = input_ids.clone()
        # Игнорируем pad_token при подсчете Loss
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return pixel_values.squeeze(), labels.squeeze()


class DonutModule(LightningModule):
    def __init__(self, processor, model, lr, dataset_path, batch_size):
        super().__init__()
        self.processor = processor
        self.model = model
        self.lr = lr
        self.dataset_path = dataset_path
        self.batch_size = batch_size

    def training_step(self, batch, batch_idx):
        pixel_values, labels = batch
        outputs = self.model(pixel_values, labels=labels)
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train_dataloader(self):
        train_dataset = DonutDataset(self.dataset_path, self.processor)
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )


def main(args):
    print(f"🔧 Инициализация обучения для датасета: {args.dataset}")

    # 1. Загрузка конфига и модели
    config = VisionEncoderDecoderConfig.from_pretrained(MODEL_REPO)
    config.encoder.image_size = IMAGE_SIZE
    config.decoder.max_length = MAX_LENGTH

    processor = DonutProcessor.from_pretrained(MODEL_REPO)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_REPO, config=config)

    # 2. Настройка токенизатора
    processor.tokenizer.pad_token = processor.tokenizer.unk_token
    # Добавляем спец-токены для наших задач
    processor.tokenizer.add_special_tokens({"additional_special_tokens": ["<s_passport>", "<s_registration>"]})
    model.decoder.resize_token_embeddings(len(processor.tokenizer))

    # 3. Настройка Lightning
    module = DonutModule(processor, model, args.lr, args.dataset, args.batch)

    # Сохраняем чекпоинты в отдельную папку для каждой модели
    checkpoint_dir = os.path.join("checkpoints", args.name)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="donut-{epoch:02d}-{train_loss:.2f}",
        save_top_k=1,  # Храним только 1 лучший чекпоинт, чтобы не забить SSD
        monitor="train_loss"
    )

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=args.epochs,
        precision="bf16-mixed",  # Идеально для RTX 5090 (BFloat16)
        callbacks=[checkpoint_callback],
        gradient_clip_val=1.0  # Защита от взрыва градиентов
    )

    # 4. Запуск
    print(f"🚀 Старт обучения модели '{args.name}' на RTX 5090...")
    trainer.fit(module)

    # 5. Сохранение финальных весов
    output_model_dir = os.path.join("models_ready", args.name)
    os.makedirs(output_model_dir, exist_ok=True)
    model.save_pretrained(output_model_dir)
    processor.save_pretrained(output_model_dir)
    print(f"✅ Модель успешно сохранена в '{output_model_dir}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение Donut OCR")
    parser.add_argument('--dataset', type=str, required=True,
                        help='Путь к папке с датасетом (где лежит metadata.jsonl)')
    parser.add_argument('--name', type=str, required=True, help='Имя модели (создаст папку с таким именем)')
    parser.add_argument('--epochs', type=int, default=10, help='Количество эпох')
    parser.add_argument('--batch', type=int, default=4, help='Размер батча (на 5090 можно 4 или 8)')
    parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate')

    args = parser.parse_args()
    main(args)