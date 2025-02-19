import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


model = "DmitrySharonov/bert_tiny2_nexign"
labels = ["B", "N", "G"]


def load_model():
    global tokenizer, model, device
    try:
        tokenizer = AutoTokenizer.from_pretrained(model)
        model = AutoModelForSequenceClassification.from_pretrained(model)
        model.eval()
        device = torch.device("cpu")
        model.to(device)
        logging.info("Модель успешно загружена")
    except Exception as e:
        logging.error(f"Ошибка при загрузке модели: {e}")
        raise e


def predict(text_input: str) -> str:
    inputs = tokenizer(
        text_input,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    predicted_class_idx = torch.argmax(logits, dim=-1).item()
    return labels[predicted_class_idx]


def process_csv(input_csv: str, output_csv: str):
    df = pd.read_csv(input_csv, sep=',')
    df["Class"] = df["MessageText"].astype(str).apply(predict)
    result_df = df[["UserSenderId", "Class"]]
    result_df.to_csv(output_csv, sep=',', index=False)
    logging.info(f"Результаты сохранены в {output_csv}")


if __name__ == "__main__":
    load_model()
    input_csv = "example_input.csv"
    output_csv = "submission.csv"
    process_csv(input_csv, output_csv)
