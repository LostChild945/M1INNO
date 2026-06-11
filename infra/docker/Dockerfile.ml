FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY src/ml/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# CmdStan requis par Prophet — cmake doit être disponible pour la compilation
RUN python -c "import cmdstanpy; cmdstanpy.install_cmdstan()" && \
    python -c "import cmdstanpy; print('[Dockerfile.ml] CmdStan OK:', cmdstanpy.cmdstan_path())"

COPY src/ ./src/

CMD ["bash", "-c", \
    "python3 /app/src/ml/simulate_data.py && \
     python3 /app/src/ml/train_xgboost.py && \
     python3 /app/src/ml/train_prophet.py"]
