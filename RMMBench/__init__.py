import os
import sys
import logging

package_root = os.path.dirname(os.path.abspath(__file__))


for env_var_name in ('RMMBENCH_ROOT'):
    os.environ.setdefault(env_var_name, package_root)
