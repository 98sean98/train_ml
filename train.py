import os
from time import sleep

import numpy as np
import mlflow


# if running locally, make sure that the environment vairables are set before calling mlflow functions
# os.environ["MLFLOW_TRACKING_USERNAME"] = "98sean98/test"
# os.environ["MLFLOW_TRACKING_PASSWORD"] = "generated token"

# mlflow.set_tracking_uri('https://community.mlflow.deploif.ai')
# mlflow.set_tracking_uri('http://localhost:8080')
mlflow.set_tracking_uri('https://strong-dragon-10.loca.lt')
mlflow.set_experiment('98sean98/test')


with mlflow.start_run() as run:
    print('mlflow run id', run.info.run_id)

    a = np.random.rand(4,4)
    print('a:', a)

    script_dir = os.path.dirname(__file__)
    output_path = os.path.join(script_dir, 'artifacts/output.txt')

    np.savetxt(output_path, a, fmt='%4.4f')

    avg = np.average(a)
    print('avg:', avg)

    mlflow.log_metric('avg', avg)
