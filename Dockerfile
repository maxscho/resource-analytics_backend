FROM python:3.10
WORKDIR /app
RUN apt-get update && apt-get install graphviz -y
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN mkdir -p files
ENV API_URL=http://localhost:9090
CMD ["./run.sh"]
