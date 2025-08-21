from typing import Callable, Optional

from joblib import Parallel, delayed
from tqdm import tqdm


class ProgressParallel(Parallel):
    def __init__(self, total: Optional[int] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total = total

    def __call__(self, *args, **kwargs):
        with tqdm(total=self.total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


def run_in_parallel(function: Callable, data: list, n_jobs=15) -> list:
    tasks = (delayed(function)(item) for item in data)
    results = ProgressParallel(n_jobs=n_jobs, total=len(data))(tasks)
    return results