FROM python:3.9-slim

# Instala dependencias de sistema necesarias para OpenCV y multimedia
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    ffmpeg \
    gcc \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Crea directorio de la app
WORKDIR /app

# Copia los archivos al contenedor
COPY . .

# Instala dependencias de Python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expone el puerto usado por gunicorn
EXPOSE 8080

# Comando para iniciar la app
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:server"]
