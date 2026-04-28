import runpy

from pricing__epac.src.machine_learning.orchestration.watcher import *  # noqa: F401,F403

if __name__ == "__main__":
    runpy.run_module("pricing__epac.src.machine_learning.orchestration.watcher", run_name="__main__")
