FROM python:3.11-slim
WORKDIR /app
COPY classifier/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY src/model.py src/preprocess.py ./
COPY classifier/app.py classifier/model.pth ./
ENV PORT=8000 POLICY_URL=http://policy:8001/decide
EXPOSE 8000
CMD ["python", "app.py"]
