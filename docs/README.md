# FraudLens model documentation

Model cards for the two models this project trains and serves. Each card
reports what its own training run actually measured (with the exact command
to reproduce those numbers), not aspirational or target figures. Both models
are now trained on real Kaggle competition data (IEEE-CIS Fraud Detection
for the fraud model, Home Credit Default Risk for the credit scorecard); the
synthetic generators each card still documents remain available as a fast,
no-download fallback for local iteration, but are no longer what these
cards' own reported numbers come from.

- [`model_card_fraud_model.md`](model_card_fraud_model.md): the fraud
  transaction scoring model (`fraudlens_fraud_model` in the MLflow Model
  Registry), served at `POST /score/fraud`.
- [`model_card_credit_scorecard.md`](model_card_credit_scorecard.md): the
  credit risk scorecard (`fraudlens_credit_scorecard` in the MLflow Model
  Registry), served at `POST /score/credit`, including its automated PSI
  drift monitoring.

Both cards were written against a real, reproducible training run of
`python -m src.serving.train_serving_models` pointed at the real, downloaded
Kaggle files via `--transaction-csv`/`--identity-csv`/`--credit-csv`; each
card's own Section 7 or 8 gives the exact command and explains why that run
reproduces exactly on a re-run against the same files.
