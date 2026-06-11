FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY src/ml/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# prophet 1.1.5 bundles cmdstan-2.33.1 without a makefile,
# but cmdstanpy >=1.2 requires it in validate_cmdstan_path().
# Creating an empty makefile satisfies the check — prophet uses
# its pre-compiled binary and never invokes make anyway.
RUN touch /usr/local/lib/python3.11/site-packages/prophet/stan_model/cmdstan-2.33.1/makefile

COPY src/ ./src/

CMD ["bash", "-c", \
    "python3 /app/src/ml/simulate_data.py && \
     python3 /app/src/ml/train_xgboost.py && \
     python3 /app/src/ml/train_prophet.py"]
