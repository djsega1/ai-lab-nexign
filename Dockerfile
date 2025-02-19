FROM ubuntu:20.04
RUN apt-get update && apt-get install -y python3.12 python3.12-dev
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "main.py"]