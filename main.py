import asyncio
import gradio as gr
import concurrent.futures
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

tokenizer = None
model = None
device = None
checkpoint_path = "./results/checkpoint-epoch-1"
labels = ["negative", "neutral", "positive"]

NUM_WORKERS = 4

executor = concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS)


def load_model():
    global tokenizer, model, device
    try:
        tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
        model.eval()
        device = torch.device("cpu")
        model.to(device)
        logging.info("Successfully loaded model")
    except Exception as e:
        logging.error(f"Error in loading model: {e}")
        raise e


async def predict(text_input: str) -> str:
    logging.info(f"Text for prediction: {text_input}")

    def blocking_predict(text: str) -> str:
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
            inputs = {key: value.to(device) for key, value in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits

            predicted_class_idx = torch.argmax(logits, dim=-1).item()
            predicted_label = labels[predicted_class_idx]
            logging.info(f"Predicted class: {predicted_class_idx}")
            logging.info(f"Predicted label: {predicted_label}")
            return predicted_label
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            raise e

    loop = asyncio.get_running_loop()
    try:
        predicted_label = await loop.run_in_executor(executor, blocking_predict, text_input)
        return predicted_label
    except Exception as e:
        logging.error(f"Executor run failed: {e}")
        raise e
    

async def clear_all():
    return [None] * 4


with gr.Blocks(theme=gr.themes.Soft(), title="Анализатор тональности") as app:
    gr.Markdown("## 📊 Анализ тональности текстов")
    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="Введите текст (разделяйте Enter)",
                placeholder="Пример:\nОтличный сервис!\nУжасное качество...",
                lines=5
            )
            # file_input = gr.File(label="Или загрузите Excel файл", type="filepath")
            with gr.Row():
                submit_btn = gr.Button("Анализировать", variant="primary")
                clear_btn = gr.Button("Очистить", variant="secondary")
        with gr.Column(scale=1):
            output_text = gr.Textbox(label="Результаты", interactive=False)
            # download_btn = gr.DownloadButton(
            #     label="📥 Скачать результаты",
            #     visible=False
            # )
    submit_btn.click(
        fn=predict,
        #inputs=[text_input, file_input],
        inputs=[text_input,],
        # outputs=[output_text, download_btn],
        outputs=[output_text,],
        api_name="predict",
        concurrency_limit=4,
    )
    clear_btn.click(
        fn=clear_all,
        # outputs=[text_input, file_input, output_text, download_btn],
        outputs=[text_input, output_text,],
        queue=False
    )
app.queue(
    max_size=20,
    api_open=False
)


async def main():
    try:
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
    except Exception as e:
        logging.error(e)
        raise e


if __name__ == "__main__":
    asyncio.run(main())
