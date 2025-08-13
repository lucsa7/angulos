FROM python:3.9-slim

# Evitar pyc y logs bufferizados
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# SO deps mínimas para OpenCV/MediaPipe (sin libgl1-mesa-glx)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    libsm6 \
    libxrender1 \
    libxext6 \
    libx11-6 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxi6 \
    libxtst6 \
    libxrandr2 \
    libgbm1 \
    libdrm2 \
    libnss3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdbus-1-3 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libasound2 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar deps de Python con caché óptimo
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copiar el resto del proyecto
COPY . .

# Railway suele definir PORT; usa 8080 por defecto si no está
EXPOSE 8080
CMD ["sh","-c","gunicorn --bind 0.0.0.0:${PORT:-8080} app:server"]
