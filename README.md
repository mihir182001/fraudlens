\# FraudLens



A credit and fraud risk analytics platform: an end-to-end implementation of practical methods used in credit risk and fraud analytics, from raw transaction and application data through to model training, evaluation, serving, monitoring, and audit logging.



The project combines real Kaggle datasets with reproducible synthetic experiments. The production-facing components include a fraud transaction scoring model and a credit risk scorecard, both served through a FastAPI application with MLflow model versioning, an audit log, decision policies, and automated drift monitoring.



\---



\## What is served today



| | Fraud Transaction Scoring | Credit Risk Scorecard |

|---|---|---|

| Real training data | IEEE-CIS Fraud Detection (Kaggle) | Home Credit Default Risk (Kaggle) |

| Rows / realized positive rate | 590,540 transactions, 3.50% fraud | 307,511 applications, 8.07% default |

| Model | XGBoost + SMOTE, 417 auto-detected numeric/boolean features | Logistic regression on WoE-encoded features, 12 features selected by Information Value, converted into a points scorecard |

| Test AUC-ROC | 0.8875 | 0.7374 |

| Test AUC-PR | 0.4948 | 0.2040 |

| KS statistic | 0.6260 | 0.3649 |

| Gini coefficient | 0.7749 | 0.4747 |

| Served at | `POST /score/fraud` | `POST /score/credit` |

| MLflow Model Registry | `fraudlens\_fraud\_model`, alias `champion`, version 4 | `fraudlens\_credit\_scorecard`, alias `champion`, version 4 |



These results come directly from real training runs using the downloaded Kaggle datasets.



Detailed methodology, validation results, limitations, and exact reproduction commands are documented in the model cards:



\- \[`docs/model\_card\_fraud\_model.md`](docs/model\_card\_fraud\_model.md)

\- \[`docs/model\_card\_credit\_scorecard.md`](docs/model\_card\_credit\_scorecard.md)

\- \[`docs/README.md`](docs/README.md)



\---



\## How it works



\### Fraud transaction scoring



The fraud pipeline uses the IEEE-CIS Fraud Detection dataset, combining transaction and identity information using a schema-aware left join.



A left join is used because not every transaction has a corresponding identity record. Using an inner join would silently remove transactions without identity information.



Feature engineering creates point-in-time-correct transaction and card-usage features. A feature is only allowed to use information that existed before the transaction being scored.



Examples include:



\- Transaction velocity

\- Card-usage aggregates

\- Frequency-based features

\- Transaction-level behavioural signals



Every feature is protected by an explicit leakage guard test.



Training uses a chronological train/test split rather than a random split because fraud detection is inherently time-dependent. This prevents the validation process from mixing future observations into the training period.



The production fraud model uses XGBoost with SMOTE applied only inside a resampling-aware training pipeline, ensuring that the test set is never resampled.



The real-data training run uses a fixed `random\_state=42` for the relevant modelling and resampling components to maintain reproducibility.



\### Credit risk scorecard



The credit risk component uses the Home Credit Default Risk dataset.



The real application dataset is reduced to an 18-field schema matching the serving API request structure.



Weight of Evidence (WoE) binning and Information Value (IV) are used to identify predictive variables. The final model uses 12 selected features.



Each selected feature is divided into five WoE bins fitted using the training data only.



The `DAYS\_EMPLOYED` value `365243`, which represents a special placeholder for unemployed or retired applicants in the dataset, is handled as its own explicit category rather than being treated as a genuine employment duration.



A logistic regression model is trained on the WoE-transformed features and converted into a points-based scorecard using the standard Offset/Factor transformation.



This allows the final credit score to be calculated through fixed point contributions rather than requiring a live model calculation for every score.



The real-data train/test split uses a fixed `random\_state=0`, documented in the credit model card.



\---



\## Model Performance



\### Fraud Transaction Model



| Metric | Result |

|---|---:|

| AUC-ROC | 0.8875 |

| AUC-PR | 0.4948 |

| KS Statistic | 0.6260 |

| Gini Coefficient | 0.7749 |



\### Credit Risk Scorecard



| Metric | Result |

|---|---:|

| AUC-ROC | 0.7374 |

| AUC-PR | 0.2040 |

| KS Statistic | 0.3649 |

| Gini Coefficient | 0.4747 |



These metrics are based on internal validation splits of the public Kaggle training datasets and are not official competition leaderboard results.



\---



\## Serving



Both trained models are loaded once when the FastAPI application starts and are not refitted during individual API requests.



The application exposes two scoring endpoints.



\### Fraud scoring



```http

POST /score/fraud

```



\### Credit scoring



```http

POST /score/credit

```



\### Decision policy



A separate decision policy layer converts model outputs into:



\- `approve`

\- `review`

\- `decline`



The decision policy is implemented separately from the machine learning models so that scoring logic and business decision logic can be tested independently.



Decision thresholds are calculated using the respective model's training-set score distribution.



\### Audit logging



Every scored decision is written to a durable SQLite audit database.



The audit log provides a queryable record of scoring activity and decisions.



The dashboard reads directly from this audit database and the MLflow tracking database, allowing model monitoring and audit information to remain available independently of the API process.



The current implementation uses SQLite as a lightweight project-level audit store. It is not intended to provide production-grade tamper-evident or append-only storage.



\### Model monitoring



The credit risk scorecard includes Population Stability Index (PSI) monitoring.



The monitoring process compares scores from a new batch of applications against the saved reference distribution from the model's validation data.



```

New Application Batch

&#x20;       |

&#x20;       v

Credit Risk Scorecard

&#x20;       |

&#x20;       v

Generate Credit Scores

&#x20;       |

&#x20;       v

Compare with Reference Distribution

&#x20;       |

&#x20;       v

PSI Calculation

&#x20;       |

&#x20;       v

Drift Detection

```



The PSI monitoring implementation is standalone and automatable.



The monitoring script can be used as part of a scheduled job or CI workflow and exits with a non-zero status when significant distributional shift is detected.



\### MLflow model management



Every training run registers the resulting model as a new version in the MLflow Model Registry.



The registry maintains:



\- Model versions

\- Training run information

\- Model metadata

\- Champion model alias

\- Traceability between model artifacts and training runs



The API uses the registered model artifacts, providing a clear link between a served model and the training run that produced it.



Current registered models:



\- `fraudlens\_fraud\_model`

\- `fraudlens\_credit\_scorecard`



Both currently use the `champion` alias.



\---



\## Research Experiments



The project also contains several research modules exploring additional techniques using disclosed synthetic datasets.



These experiments are kept separate from the two real-data serving models.



The synthetic datasets are used to create controlled environments where specific methods can be evaluated and reproduced without making claims about real-world fraud, credit, or marketing behaviour.



\### Graph Neural Networks



A heterogeneous transaction graph connects transaction nodes with shared entities such as:



\- Cards

\- Addresses

\- Email domains



A GraphSAGE-based model is used to investigate whether relationships between transactions can provide additional signals for identifying coordinated fraud patterns.



This approach represents relationships between entities that traditional row-based tabular models cannot directly capture.



\### Uplift Modelling



The project includes individual-level treatment effect modelling using:



\- S-Learner

\- T-Learner



The models are evaluated using the Qini coefficient.



The synthetic data generator provides the underlying treatment effects, allowing estimated uplift to be compared with an oracle upper bound.



\### Active Learning



The active learning pipeline uses pool-based uncertainty sampling to determine which unlabeled transactions would provide the greatest value for human investigation.



Different sampling strategies are evaluated against the project's synthetic fraud dataset.



Entropy-based sampling reached the fully labelled training set's test AUC-PR of 0.8305 using 225 of 4,000 labels (5.6% of the available labels).



Random sampling did not reach the same result within the same labeling budget.



\### Transfer Learning



The project compares different approaches for transferring fraud detection models between domains:



\- Zero-shot transfer

\- XGBoost warm-start training

\- Training from scratch



The experiments evaluate performance across different target-domain labelling budgets.



The synthetic experiments also demonstrate the impact of domain shift. Under the simulated pattern shift, zero-shot transfer AUC-PR decreased from 0.9658 to 0.6058.



This experiment is intended to demonstrate how changes in the underlying fraud patterns can affect model transferability.



\---



\## Project Structure



```

fraudlens/

&#x20; src/

&#x20;   data/              loaders, mergers, synthetic generators

&#x20;   eda/               exploratory analysis

&#x20;   features/          point-in-time feature engineering

&#x20;   models/            baseline models, advanced models, evaluation metrics

&#x20;   graph/             heterogeneous GraphSAGE fraud model

&#x20;   uplift/            S-Learner and T-Learner models

&#x20;   active\_learning/   uncertainty sampling and evaluation

&#x20;   transfer\_learning/ domain transfer experiments

&#x20;   credit\_risk/       WoE encoding, scorecard modelling, PSI drift monitoring

&#x20;   serving/           model training, decision policy, audit logging,

&#x20;                      MLflow registry helpers

&#x20;   api/               FastAPI application

&#x20;   dashboard/         Streamlit monitoring dashboard

&#x20; tests/               automated test suite

&#x20; docs/

&#x20;   model\_card\_fraud\_model.md

&#x20;   model\_card\_credit\_scorecard.md

&#x20;   README.md

&#x20; Dockerfile, docker-compose.yml, .dockerignore

&#x20; .github/workflows/ci.yml

&#x20; requirements.txt

&#x20; README.md

```



\## Setup



Create a virtual environment:



```bash

python -m venv .venv

```



Windows:



```bash

.venv\\Scripts\\activate

```



macOS / Linux:



```bash

source .venv/bin/activate

```



Install the required dependencies:



```bash

pip install -r requirements.txt

```



\## Running the Tests



Run the complete test suite:



```bash

python -m pytest tests/ -v

```



The test suite contains 306 tests and covers the major components of the project.



Testing includes:



\- Data processing

\- Feature engineering

\- Leakage prevention

\- Model reproducibility

\- Credit scorecard calculations

\- WoE transformations

\- Points score conversion

\- API endpoints

\- Decision policies

\- Audit logging



The test suite includes dedicated leakage guard tests for the point-in-time fraud features and reproducibility tests for the credit scorecard transformations.



API tests exercise both scoring endpoints end-to-end using temporary model and audit database locations.



\## Training the Serving Models



\### Synthetic Data



For local development and experimentation, the models can be trained using the project's synthetic data:



```bash

python -m src.serving.train\_serving\_models

```



This does not require downloading the Kaggle datasets.



\### Real Kaggle Data



To reproduce the reported real-data results, download the required datasets and provide their local paths:



```bash

python -m src.serving.train\_serving\_models \\

&#x20; --transaction-csv path/to/train\_transaction.csv \\

&#x20; --identity-csv path/to/train\_identity.csv \\

&#x20; --credit-csv path/to/application\_train.csv

```



The fraud model requires both:



\- `--transaction-csv`

\- `--identity-csv`



The credit model can be trained independently using:



\- `--credit-csv`



The training process generates:



\- `models/fraud\_model.joblib`

\- `models/credit\_scorecard.joblib`



and registers the trained models with MLflow.



Each new training run creates a new MLflow model version and promotes the resulting model to the `champion` alias.



\## Running the API



Start the FastAPI application:



```bash

uvicorn src.api.main:app --reload

```



The interactive OpenAPI documentation is available at:



`http://127.0.0.1:8000/docs`



The API exposes:



\- `POST /score/fraud`

\- `POST /score/credit`



\## Running the Dashboard



Start the Streamlit dashboard in a separate terminal:



```bash

streamlit run src/dashboard/app.py

```



The dashboard reads from the local MLflow tracking database and audit log.



It provides visibility into:



\- Model comparisons

\- Prediction history

\- Decision outcomes

\- PSI drift history

\- Model information



The dashboard is intentionally decoupled from the API process so that monitoring information remains accessible even when the API is not currently running.



\## Docker



Build the Docker image:



```bash

docker build -t fraudlens .

```



Run the application using Docker Compose:



```bash

docker compose up

```



\## CI/CD



The GitHub Actions workflow runs automatically on pushes and pull requests.



The CI pipeline:



\- Installs the project dependencies

\- Runs the complete pytest suite

\- Builds the Docker image

\- Trains small smoke-test model artifacts

\- Starts the containerized API

\- Performs an API health check



This provides automated validation of both the Python codebase and the containerized serving environment.



\## Reproducibility



The project follows several practices to make experiments and model training reproducible:



\- Fixed random seeds where appropriate

\- Chronological validation for fraud modelling

\- Explicit leakage prevention tests

\- Train-only fitting for feature transformations

\- Reproducible WoE and scorecard transformations

\- MLflow model versioning

\- Automated testing

\- Docker-based execution

\- Documented model cards

\- Separation of real-data models and synthetic research experiments



\## Data Sources



\### IEEE-CIS Fraud Detection



Used for the fraud transaction scoring model.



The dataset contains transaction and identity information used to build and evaluate the fraud detection pipeline.



\### Home Credit Default Risk



Used for the credit risk scorecard.



The dataset contains loan application information used to develop the WoE-based credit scoring model.



\### Synthetic Data



Synthetic datasets are used for research experiments involving:



\- Graph neural networks

\- Uplift modelling

\- Active learning

\- Transfer learning



Synthetic experiment results are kept separate from the real-data model results.



\## Known Limitations



\- The real-data metrics are based on this project's internal validation splits of the public Kaggle training datasets and are not official competition private leaderboard results.

\- Decision thresholds and scorecard cutoffs are illustrative project-level choices and have not been calibrated against real business costs, regulatory requirements, or production approval policies.

\- The synthetic research experiments have not been validated against real production data.

\- Results from synthetic experiments should not be interpreted as claims about real-world fraud, credit, or marketing behaviour.

\- The audit log uses a mutable SQLite database and is not a tamper-evident or append-only production audit system.

\- The project does not use real customer, banking, or personally identifiable financial data.

\- Production deployment infrastructure such as managed databases, distributed feature stores, cloud orchestration, and enterprise access controls is outside the scope of this project.



\## Project Scope



FraudLens demonstrates an end-to-end workflow covering:



```

Data

&#x20; |

&#x20; v

Exploration

&#x20; |

&#x20; v

Feature Engineering

&#x20; |

&#x20; v

Model Development

&#x20; |

&#x20; v

Model Evaluation

&#x20; |

&#x20; v

Model Registry

&#x20; |

&#x20; v

API Serving

&#x20; |

&#x20; v

Decision Policy

&#x20; |

&#x20; v

Audit Logging

&#x20; |

&#x20; v

Monitoring

&#x20; |

&#x20; v

CI/CD

```



The project brings together traditional credit risk modelling, machine learning-based fraud detection, advanced research techniques, model serving, monitoring, and reproducible engineering practices in a single platform.



\## Disclaimer



FraudLens is an educational and portfolio project designed to demonstrate practical credit risk, fraud analytics, machine learning engineering, and model-serving workflows.



The models, scores, thresholds, and decision policies are not intended to be used for real financial, lending, fraud investigation, or credit approval decisions.

