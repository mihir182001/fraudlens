# Deploying the FraudLens scoring API

This project can be deployed as a live, public web service using the
render.yaml Blueprint at the repository root, which deploys the FastAPI
scoring API (src/api/main.py) to Render's free tier. This document
explains exactly what gets deployed, what does not, and why, so the
deployed service is not mistaken for more than it actually is.

## What is deployed

Only the scoring API: POST /score/fraud, POST /score/credit, and the
interactive OpenAPI page at /docs. The Streamlit monitoring dashboard
(src/dashboard/app.py) is deliberately not deployed alongside it; see
"Why the dashboard is not deployed here" below.

## The deployed model is a synthetic-data demo, not the real-data model

The Dockerfile this project deploys from bakes a model into the image at
build time by running python -m src.serving.train_serving_models
--skip-registry with its default arguments, which trains on this
project's own synthetic data generators (5,000 synthetic transactions for
the fraud model, 8,000 synthetic applications for the credit scorecard),
not the real Kaggle data. This is a deliberate, necessary choice: the real
IEEE-CIS and Home Credit training files are large, were downloaded once by
hand to a local machine, and are not something a public build server
should fetch or store. The result is that the deployed API's /score
responses come from a demonstrably different model instance than the one
this project's model cards report real-data numbers for.

The real-data performance numbers (AUC-ROC 0.8875 for fraud, 0.7374 for
credit, and the rest) in the top-level README.md and in
docs/model_card_fraud_model.md / docs/model_card_credit_scorecard.md
describe that separate, real-data training run, done locally, not whatever
is currently live at the deployed URL. The deployed service exists to show
that the scoring API runs and responds correctly end to end, not to
reproduce those specific numbers over the network.

## Why the dashboard is not deployed here

The dashboard reads directly from the same mlflow.db and audit_log.db
files the API writes to. Locally, docker-compose.yml makes this work by
bind-mounting one shared host folder into both the API container and the
dashboard container. Render gives every service its own separate
filesystem with no shared disk between services: Render's own disk
documentation states plainly that "a persistent disk is accessible by only
a single service instance" and that persistent disks are not available on
the free plan at all (https://render.com/docs/disks). Deploying the
dashboard as a second free service would therefore give it no way to see
the API's real activity; it would only ever show its own empty,
disconnected audit log and MLflow history, which would look broken to
anyone visiting it. The dashboard remains a local-only tool for now (see
the main README.md's "Running the Dashboard" section); turning it into a
second, genuinely connected deployed service would mean replacing the
local SQLite files with a real shared database, which is outside the scope
of this deployment.

## Free tier behavior worth knowing before you share the link

The following is confirmed directly from Render's own free tier
documentation (https://render.com/docs/free), not assumed:

- A free web service spins down after 15 minutes with no inbound traffic.
  The first request after that takes about one minute to wake it back up,
  during which Render shows a loading page to the visitor.
- The filesystem is ephemeral: any file the running container writes,
  including this project's audit_log.db, is lost every time the service
  redeploys, restarts, or spins down from inactivity. Every scored
  decision made against the deployed API during one period of uptime
  disappears once it spins down; this is expected, not a bug, and is a
  direct consequence of the free plan having no persistent disk at all.
- Each Render workspace gets 750 free instance hours per calendar month,
  shared across every free service in that workspace.

None of this affects correctness of a single request/response; it only
means the deployed API is a live demo, not a durable, always-on service
with a persistent audit trail.

## Deploying it

1. Make sure render.yaml, the updated Dockerfile, and this file are
   pushed to the main branch of the GitHub repository.
2. Go to the Render dashboard (https://dashboard.render.com) and choose
   New > Blueprint.
3. Connect the fraudlens GitHub repository. Render detects render.yaml
   automatically and shows the one service it defines, fraudlens-api, on
   the free plan.
4. Click Apply. Render builds the Docker image (this includes the
   synthetic-data training step described above, so the first build takes
   longer than a plain dependency install) and starts the service.
5. Once the build finishes, Render shows a public URL of the form
   https://fraudlens-api-xxxx.onrender.com. Open
   https://fraudlens-api-xxxx.onrender.com/docs to confirm the API is
   live and to try both scoring endpoints interactively.
6. If the first request after a period of inactivity is slow, that is the
   free tier's documented spin-up delay described above, not an error.

## Known limitations of this deployment

- Serves a synthetic-data demo model, not the real-data model the project's
  model cards report numbers for.
- No audit log history or MLflow version history persists across a
  redeploy, restart, or idle spin-down, since the free plan has no
  persistent disk.
- The dashboard is not deployed; it remains a local-only tool.
- The free plan's 15-minute idle spin-down means the first request after a
  quiet period is slow (about one minute) while the service wakes up.