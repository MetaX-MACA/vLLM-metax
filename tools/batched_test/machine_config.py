# SPDX-License-Identifier: Apache-2.0
from datetime import datetime

# ------------- user-tweakable defaults -------------
NUM_GPUS = 8  # total GPUs on machine
MODEL_CONFIG_FILE = "model_config/model_jiajia.yaml"  # model configuration file
MODEL_FUNCTIONAL_TEST_RESULT_DIR = f"/workspace/model_functional_test_{datetime.now().strftime('%Y_%m_%d_%H_%M')}"  # where to store test results
# ---------------------------------------------------
