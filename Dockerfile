FROM python:3.12-slim

WORKDIR /app

# 仅复制必要文件
COPY requirements.txt .
COPY main.py .

# 安装依赖（不缓存，减少镜像层体积）
RUN pip install --no-cache-dir -r requirements.txt

# 运行时下载模型文件（不打包到镜像中）
CMD ["sh", "-c", "python main.py"]
