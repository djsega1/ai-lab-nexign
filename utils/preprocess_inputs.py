import pandas as pd
import logging


def preprocess_inputs(text_input: str, file_input: str) -> pd.DataFrame:
    """Обработка и объединение входных данных"""
    texts = []

    # Обработка Excel файла
    if file_input:
        try:
            df = pd.read_excel(file_input)
            text_col = next(col for col in df.columns if "text" in col.lower())
            texts.extend(df[text_col].astype(str).tolist())
        except Exception as e:
            logging.error(f"File processing error: {str(e)}")
            raise ValueError(
                "Некорректный формат файла. Убедитесь в наличии колонки с текстом"
            )

    # Обработка текстового ввода
    if text_input:
        texts.extend([t.strip() for t in text_input.split("\n") if t.strip()])

    if not texts:
        raise ValueError("Введите текст или загрузите файл")

    return pd.DataFrame({"text": texts})
