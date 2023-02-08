FROM python:3.8-slim
RUN apt-get update && apt-get install -y \
&& apt-get clean
WORKDIR /app
COPY api.py .env model.h5 requirements.txt ./
RUN pip install --no--cache-dir -r requirements.txt
CMD uvicorn api:app --host 0.0.0.0 --port 80 --workers6