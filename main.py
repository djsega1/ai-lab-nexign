import asyncio
import os
import gradio as gr
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
from datasets import Dataset
from utils import preprocess_inputs

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)


label_mapping = {
    "B": "negative",
    "N": "neutral",
    "G": "positive"
}
label_to_id = {"negative": 0, "neutral": 1, "positive": 2}
id_to_label = {0: "negative", 1: "neutral", 2: "positive"}

model: str = os.getenv('MODEL', 'DmitrySharonov/mini_test_bert')
tokenizer: str = os.getenv('TOKENIZER', 'DmitrySharonov/mini_test_bert')
device: str = os.getenv('DEVICE', 'cpu')
tokenizer = None

NUM_WORKERS: int = int(os.getenv('NUM_WORKERS', 4))
BATCH_SIZE: int = int(os.getenv('BATCH_SIZE', 32))
MAX_LENGTH: int = int(os.getenv('MAX_LENGTH', 512))
GRADIO_SERVER_NAME: str = os.getenv('GRADIO_SERVER_NAME', '0.0.0.0')
GRADIO_SERVER_PORT: int = int(os.getenv('GRADIO_PORT', 8080))
GRADIO_THREADS: int = int(os.getenv('GRADIO_THREADS', 40))
GRADIO_QUEUE_MAX_SIZE: int = int(os.getenv('GRADIO_QUEUE_MAX_SIZE', 50))
DEBUG: bool = os.getenv('DEBUG', False)


def load_model():
    global tokenizer, model, device
    try:
        tokenizer = AutoTokenizer.from_pretrained(model)
        model = AutoModelForSequenceClassification.from_pretrained(model, num_labels=3)
        model.eval()
        device = torch.device(device)
        model.to(device)
        logging.info('Successfully loaded model')
    except Exception as e:
        logging.error(f'Error in loading model: {e}')
        raise e


def blocking_predict(
    model, device, df: pd.DataFrame, tokenized_dataset: Dataset,
) -> str:
    try:
        inputs = {
            'input_ids': torch.tensor(tokenized_dataset['input_ids']).to(device),
            'attention_mask': torch.tensor(tokenized_dataset['attention_mask']).to(
                device
            ),
        }

        with torch.no_grad():
            outputs = model(**inputs).logits
            probabilities = torch.softmax(outputs.float(), dim=1).cpu().numpy()

        # Постобработка с прямым использованием вероятностей
        df['positive_prob'] = probabilities[
            :, 1
        ]  # Берем только вероятность позитивного класса
        positive_percent = (df['positive_prob'].mean() * 100).round(2)

        result_text = f'Общий позитивный тон: {positive_percent}%\n\n' + '\n'.join(
            f'{row['text']} → Позитивность: {row['positive_prob']:.2%} '
            f'{'🔵' if row['positive_prob'] > 0.7 else '🟢' if row['positive_prob'] > 0.4 else '🟡' if row['positive_prob'] > 0.2 else '🔴'}'
            for _, row in df.iterrows()
        )

        # Сохранение результатов
        output_path = 'results.csv'
        df.to_csv(output_path, index=False)

        return result_text, output_path
    except Exception as e:
        logging.error(f'Prediction error: {str(e)}')
        raise gr.Error(f'Ошибка обработки: {str(e)}')


def predict(text_input: str, file_input: str):
    logging.info(f'Text for prediction: {text_input}')
    df = preprocess_inputs(text_input, file_input)
    dataset = Dataset.from_pandas(df)

    def tokenize_fn(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors='pt',
        )

    tokenized_dataset = dataset.map(tokenize_fn, batched=True, batch_size=BATCH_SIZE)
    # loop = asyncio.get_running_loop()
    try:
        result_text, output_path = blocking_predict(model, device, df, tokenized_dataset)
        logging.info(f'Finished prediction for text: {text_input}')
        return result_text, output_path
    except Exception as e:
        logging.error(f'Executor run failed: {e}')
        raise e


async def clear_all():
    return [None] * 4


# Создание интерфейса
with gr.Blocks(theme=gr.themes.Soft(), title='Анализатор тональности') as app:
    gr.Markdown('## 📊 Анализ тональности текстов')

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label='Введите текст (разделяйте Enter)',
                placeholder='Пример:\nОтличный сервис!\nУжасное качество...',
                lines=5,
            )
            file_input = gr.File(label='Или загрузите Excel файл', type='filepath')

            with gr.Row():
                submit_btn = gr.Button('Анализировать', variant='primary')
                clear_btn = gr.Button('Очистить', variant='secondary')

        with gr.Column(scale=2):
            output_text = gr.Textbox(label='Результаты', interactive=False)
            download_btn = gr.DownloadButton(
                label='📥 Скачать результаты',
                # visible=False
            )

    # Обработчики событий с настройкой concurrency_limit
    submit_btn.click(
        fn=predict,
        inputs=[text_input, file_input],
        outputs=[output_text, download_btn],
        api_name='predict',
        concurrency_limit=NUM_WORKERS,  # Устанавливаем лимит параллельных запросов здесь
    )

    clear_btn.click(
        fn=clear_all,
        outputs=[text_input, file_input, output_text, download_btn],
        queue=False,
    )

# Новая конфигурация очереди
app.queue(max_size=GRADIO_QUEUE_MAX_SIZE, api_open=False)


async def main():
    load_model()
    app.launch(
        server_name=GRADIO_SERVER_NAME,
        server_port=GRADIO_SERVER_PORT,
        debug=DEBUG,
        max_threads=GRADIO_THREADS,
        show_error=True,
        share=False,
    )
    logging.info('Started gradio app.')


if __name__ == '__main__':
    asyncio.run(main())
