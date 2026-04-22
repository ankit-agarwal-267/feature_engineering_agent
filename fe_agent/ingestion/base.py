from typing import Protocol, Any, Union
from fe_agent.profiler.semantic_types import RawDataFrame

class AbstractDataLoader(Protocol):
    def load(self, source: str | dict, **kwargs: Any) -> RawDataFrame:
        """
        Loads data from a source and returns a raw DataFrame.
        """
        ...
        
    def supports(self, source: str | dict) -> bool:
        """
        Returns True if the loader supports the given source.
        """
        ...
