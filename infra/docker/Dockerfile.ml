FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY src/ml/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# CmdStan requis par Prophet
RUN python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"

COPY src/ ./src/

CMD ["bash", "-c", \
    "python3 /app/src/ml/simulate_data.py && \
     python3 /app/src/ml/train_xgboost.py && \
     python3 /app/src/ml/train_prophet.py"]
