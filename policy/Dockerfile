FROM python:3.11-slim
WORKDIR /app
COPY policy/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY policy/app.py .
ENV PORT=8001
EXPOSE 8001
CMD ["python", "app.py"]
