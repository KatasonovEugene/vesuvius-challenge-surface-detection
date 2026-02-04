from abc import abstractmethod
from typing import final


class BaseMetric:
    """
    Base class for all metrics
    """

    def __init__(self, name=None, *args, **kwargs):
        """
        Args:
            name (str | None): metric name to use in logger and writer.
        """
        self.name = name if name is not None else type(self).__name__

    @abstractmethod
    def __call__(self, **batch):
        """
        Defines metric calculation logic for a given batch.
        Can use external functions (like TorchMetrics) or custom ones.
        """
        raise NotImplementedError()


    def getKeys(self):
        """
        Returns list of metric keys if __call__ returns dict
        """
        return [""]


    @final
    def keys_full_list(self):
        return [self.name + "_" + key if key else self.name for key in self.getKeys()]
