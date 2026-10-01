"""
conftest.py, shared pytest configuration for this project's whole suite.

DO_NOT_TRACK is set here, once, for every test in this project, rather than
relying on whoever runs pytest to remember an environment variable: any
test that calls mlflow.set_tracking_uri, mlflow.start_run, or
mlflow.log_metric (most of tests/test_model_registry.py,
tests/test_dashboard.py, and tests/test_psi_monitoring.py) triggers an
outbound telemetry call MLflow makes on its own. On a network that blocks
that specific endpoint, this project's own development sandbox does, and
some corporate networks do too, that call's own retry logic was confirmed
directly to push a 286-test run past a 2-minute timeout. Setting
DO_NOT_TRACK=1 disables it; nothing this project's own tests check depends
on MLflow's telemetry, so this has no effect on what the suite actually
verifies, and it is a harmless no-op on a network where that endpoint was
never blocked in the first place.

setdefault, not a plain assignment, so an explicit DO_NOT_TRACK already
set in the environment before pytest starts is left alone.
"""

import os

os.environ.setdefault("DO_NOT_TRACK", "1")
