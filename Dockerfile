FROM python:3.9-slim

# Instalar librerías del sistema necesarias para OpenCV, MediaPipe y entorno gráfico mínimo
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    libglvnd0 \
    libglx0 \
    libx11-6 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxi6 \
    libxtst6 \
    libnss3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdbus-1-3 \
    libdrm2 \
    libxrandr2 \
    libgbm1 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libasound2 \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Crear carpeta de la app
WORKDIR /app

# Copiar todo el contenido al contenedor
COPY . .

# Instalar dependencias de Python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Exponer el puerto
EXPOSE 8080

# Comando para ejecutar la app
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:server"]


