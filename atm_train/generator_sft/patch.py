# patch.py
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam

# Monkey patch the DeepSpeedCPUAdam class to avoid the ds_opt_adam error
original_init = DeepSpeedCPUAdam.__init__

def patched_init(self, *args, **kwargs):
    try:
        original_init(self, *args, **kwargs)
    except Exception as e:
        print(f"Caught exception in DeepSpeedCPUAdam.__init__: {e}")
        # Add dummy attribute to avoid AttributeError in __del__
        self.ds_opt_adam = None
        self.opt_id = None

DeepSpeedCPUAdam.__init__ = patched_init