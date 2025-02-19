FROM python:3.12-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
EXPOSE 8080
COPY . .
CMD ["python", "main.py"]