# Dockerfile, FraudLens Week 12: a single image for both the FastAPI
# scoring service (src/api/main.py) and the Streamlit monitoring dashboard
# (src/dashboard/app.py). One image rather than two separate ones because
# both services import from the same src/ package and need the exact same
# dependency set (pandas, scikit-learn, xgboost, fastapi, streamlit, ...);
# building two images would mean installing that whole stack twice,
# doubling build time and image storage for no real benefit.
# docker-compose.yml runs two containers FROM this one image, distinguished
# only by which command each container runs.
FROM python:3.11-slim

WORKDIR /app

# libgomp1 provides libgomp.so.1, which xgboost's and lightgbm's compiled,
# OpenMP-parallelized extensions link against at import time, not only
# during their own build. python:3.11-slim is Debian-based and does not
# include it by default, and importing src.models.advanced_models (which
# src.api.main imports) without it fails with "OSError: libgomp.so.1:
# cannot open shared object file: No such file or directory". This was not
# confirmed inside an actual container build (this project's own sandbox
# could not pull python:3.11-slim to build one; see the delivery message
# alongside this file for why), but it was confirmed a different way: this
# same sandbox's own Linux environment, where every prior week's xgboost
# and lightgbm tests have run cleanly, already has libgomp1 installed as a
# real package for exactly that reason, and would fail those same imports
# without it. Confirm this line is actually needed on your machine when
# you run the real docker compose build.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt is copied and installed before the rest of the source
# tree so that changing application code, which happens far more often
# than changing dependencies, does not invalidate Docker's layer cache for
# the slow, multi-hundred-megabyte pip install step.
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY src/ src/

# Trained model artifacts (models/*.joblib), and the MLflow/audit SQLite
# files, are deliberately NOT copied into the image from the host: they
# are this project's data, produced by running the training scripts
# (src/serving/train_serving_models.py, src/credit_risk/train_scorecard.py,
# and the rest) on the host, not part of the image's own code.
# docker-compose.yml bind-mounts them in at container start instead, so
# retraining a model on the host and restarting the container picks up the
# new artifact without ever rebuilding the image.
#
# The line below bakes a SYNTHETIC-data demo model into the image itself,
# for a different case docker-compose.yml does not cover: running this
# image standalone, with no host to bind-mount models/ from at all (for
# example a cloud platform that only runs the container, such as the
# render.yaml deploy target documented in docs/deployment.md). This does
# NOT change local docker-compose behavior: docker-compose.yml's own
# bind mount (./models:/app/models:ro) still overrides whatever is baked
# in here the moment the container starts, exactly as before this line
# was added. --skip-registry means this build step never touches MLflow,
# since a Docker build has no reason to register a model version. A
# container running on this baked-in default is serving a model trained
# on this project's own synthetic data generators (see
# src/data/generate_synthetic_ieee.py and generate_synthetic_credit.py),
# not the real-Kaggle-data model this project's model cards report
# numbers for; docs/deployment.md states this plainly for anyone using a
# deployment built from this image.
RUN python -m src.serving.train_serving_models --skip-registry

EXPOSE 8000 8501

# The default command runs the FastAPI service; docker-compose.yml
# overrides this command for the dashboard container, so both containers
# come from this exact same image.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]