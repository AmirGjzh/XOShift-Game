import importlib.util
import os
import sys
from typing import Callable


def load_agent(agent_path: str) -> Callable:
    """
    Dynamically load a Python file at `agent_path` which defines:
       def agent_move():
       ...
    Returns a reference to that function.
    """
    # Use the filename (without .py) as the module name so pickle/multiprocessing
    # can locate it via sys.modules when sending the function to a subprocess.
    module_name = os.path.basename(agent_path).replace(".py", "")

    # If the module is already loaded (e.g. both agents use the same file),
    # return the cached function so pickle sees the same object.
    if module_name in sys.modules:
        cached_module = sys.modules[module_name]
        if hasattr(cached_module, 'agent_move'):
            return getattr(cached_module, 'agent_move')

    spec = importlib.util.spec_from_file_location(module_name, agent_path)
    agent_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = agent_module
    try:
        spec.loader.exec_module(agent_module)
    except Exception:
        # If loading fails, remove the broken module from sys.modules
        del sys.modules[module_name]
        raise

    if not hasattr(agent_module, 'agent_move'):
        del sys.modules[module_name]
        raise ValueError(f"Agent file '{agent_path}' does not define 'agent_move' function.")

    return getattr(agent_module, 'agent_move')
