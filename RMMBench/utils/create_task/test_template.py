import random
from RMMBench.tasks.dm_task import *
from RMMBench.tasks.base_task import PrimitiveTask
from RMMBench.tasks.composite.base import ClusterTask
from RMMBench.tasks.config_manager import BenchTaskConfigManager,ClusterConfigManager
from RMMBench.utils.register import register
from RMMBench.utils.utils import flatten_list, grid_sample

@register.add_config_manager("cluster_vegetables_vs_meat")
class ClusterVegetablesVsMeatConfigManager(ClusterConfigManager):
    def __init__(self,
                 task_name,
                 num_objects=3,  # total count
                 **kwargs):
        super().__init__(task_name, num_objects, **kwargs)

    def get_instruction(self, **kwargs):
        instruction = ["Cluster the objects into two classes."]
        self.config["task"]["instructions"] = instruction


@register.add_task("cluster_vegetables_vs_meat")
class ClusterTask(ClusterTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)