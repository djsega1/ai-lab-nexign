import asyncio
import gradio as gr
import pandas as pd
import concurrent.futures
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
from datasets import Dataset
from utils import preprocess_inputs


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

tokenizer = None
model = None
device = None
checkpoint_path = "DmitrySharonov/bert_tiny2_nexign"
labels = ["negative", "neutral", "positive"]

NUM_WORKERS = 4
BATCH_SIZE = 32
MAX_LENGTH = 512

executor = concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS)


def load_model():
    global tokenizer, model, device
    try:
        tokenizer = AutoTokenizer.from_pretrained("DmitrySharonov/bert_tiny2_nexign")
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
        model.eval()
        device = torch.device("cpu")
        model.to(device)
        logging.info("Successfully loaded model")
    except Exception as e:
        logging.error(f"Error in loading model: {e}")
        raise e


def blocking_predict(model, device, df: pd.DataFrame, tokenized_dataset: Dataset) -> str:
    try:
        inputs = {
            'input_ids': torch.tensor(tokenized_dataset['input_ids']).to(device),
            'attention_mask': torch.tensor(tokenized_dataset['attention_mask']).to(device)
        }

        with torch.no_grad():
            outputs = model(**inputs).logits
            probabilities = torch.softmax(outputs.float(), dim=1).cpu().numpy()

        # Постобработка с прямым использованием вероятностей
        df['positive_prob'] = probabilities[:, 1]  # Берем только вероятность позитивного класса
        positive_percent = (df['positive_prob'].mean() * 100).round(2)

        # Очистка памяти
        del inputs, outputs
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        elif device.type == 'mps':
            torch.mps.empty_cache()

        result_text = f"Общий позитивный тон: {positive_percent}%\n\n" + "\n".join(
            f"{row['text']} → Позитивность: {row['positive_prob']:.2%} " 
            f"{'🔵' if row['positive_prob'] > 0.7 else '🟢' if row['positive_prob'] > 0.4 else '🟡' if row['positive_prob'] > 0.2 else '🔴'}"
            for _, row in df.iterrows()
        )

        
        # Сохранение результатов
        output_path = "results.xlsx"
        df.to_excel(output_path, index=False)
        
        return result_text, output_path
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise gr.Error(f"Ошибка обработки: {str(e)}")


async def predict(text_input: str, file_input: str):
    logging.info(f"Text for prediction: {text_input}")
    df = preprocess_inputs(text_input, file_input)
    dataset = Dataset.from_pandas(df)
    def tokenize_fn(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
    tokenized_dataset = dataset.map(tokenize_fn, batched=True, batch_size=BATCH_SIZE)
    loop = asyncio.get_running_loop()
    try:
        result_text, output_path = await loop.run_in_executor(
            executor,
            blocking_predict,
            model,
            device,
            df,
            tokenized_dataset
        )
        return result_text, output_path
    except Exception as e:
        logging.error(f"Executor run failed: {e}")
        raise e
    

async def clear_all():
    return [None] * 4


# Создание интерфейса
with gr.Blocks(theme=gr.themes.Soft(), title="Анализатор тональности") as app:
    gr.Markdown("## 📊 Анализ тональности текстов")
    
    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Введите текст (разделяйте Enter)",
                placeholder="Пример:\nОтличный сервис!\nУжасное качество...",
                lines=5
            )
            file_input = gr.File(label="Или загрузите Excel файл", type="filepath")
            
            with gr.Row():
                submit_btn = gr.Button("Анализировать", variant="primary")
                clear_btn = gr.Button("Очистить", variant="secondary")
        
        with gr.Column(scale=2):
            output_text = gr.Textbox(label="Результаты", interactive=False)
            download_btn = gr.DownloadButton(
                label="📥 Скачать результаты",
                #visible=False
            )

    # Обработчики событий с настройкой concurrency_limit
    submit_btn.click(
        fn=predict,
        inputs=[text_input, file_input],
        outputs=[output_text, download_btn],
        api_name="predict",
        concurrency_limit=4  # Устанавливаем лимит параллельных запросов здесь
    )
    
    clear_btn.click(
        fn=clear_all,
        outputs=[text_input, file_input, output_text, download_btn],
        queue=False
    )

# Новая конфигурация очереди
app.queue(
    max_size=20,
    api_open=False
)


async def main():
    load_model()
    app.launch(
        server_name="0.0.0.0",
        server_port=8080,
        show_error=True,
        share=False,
        debug=False,
        max_threads=40
    )
    logging.info("Started gradio app.")


if __name__ == "__main__":
    asyncio.run(main())
    # load_model()
    # app.launch(
    #     server_name="0.0.0.0",
    #     server_port=8080,
    #     show_error=True,
    #     share=False,
    #     debug=False,
    #     max_threads=40
    # )
    # logging.info("Started gradio app.")
