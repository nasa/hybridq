"""
Author: Salvatore Mandra (salvatore.mandra@nasa.gov)

Copyright © 2021, United States Government, as represented by the Administrator
of the National Aeronautics and Space Administration. All rights reserved.

The HybridQ: A Hybrid Simulator for Quantum Circuits platform is licensed under
the Apache License, Version 2.0 (the "License"); you may not use this file
except in compliance with the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""
import logging

__all__ = [
    'Function', 'map', 'starmap', 'get_n_workers', 'init', 'shutdown', 'restart'
]

# create logger
_LOGGER = logging.getLogger(__name__)
_LOGGER.setLevel(logging.INFO)
_LOGGER_CH = logging.StreamHandler()
_LOGGER_CH.setLevel(logging.DEBUG)
_LOGGER_CH.setFormatter(
    logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s'))
_LOGGER.addHandler(_LOGGER_CH)


class Function:
    """
    Make any function pickleable.

    Parameters
    ----------
    fn: callable
        Function to pickle.
    pickler: str, optional
        Valid pickle module to use.
    """
    __slots__ = ('_fn', '_pickler')

    def __init__(self, fn: callable, /, *, pickler: str = None):
        from os import environ

        # Store function
        self._fn = fn

        # Store module
        self._pickler = environ.get('HYBRIDQ_PARALLEL_PICKLER', 'dill')

    def __str__(self):
        return str(self._fn)

    def __repr__(self):
        return repr(self._fn)

    def __call__(self, *args, **kwargs):
        # Return results
        return self._fn(*args, **kwargs)

    def __getstate__(self):
        from importlib import import_module

        # Get pickler
        _pickler = import_module(self._pickler)

        # Dump state
        return self._pickler, _pickler.dumps(self._fn)

    def __setstate__(self, state):
        from importlib import import_module

        # Get pickler and state
        self._pickler, state = state

        # Import pickler
        _pickler = import_module(self._pickler)

        # Update
        self._fn = _pickler.loads(state)


class Executor:
    __slots__ = ('_executor', '_n_workers')

    def __init__(self):
        # Initialize _executor to None
        self._executor = None

        # Initialie _n_workers to zero
        self._n_workers = 0

    def __del__(self):
        # Shutdown
        self.shutdown()

    @property
    def executor(self):
        return self._executor

    @property
    def n_workers(self):
        return self._n_workers

    def shutdown(self, wait=True, *, cancel_futures=False):
        """
        See Also
        --------
        concurrent.futures.ProcessPoolExecutor.shutdown
        """

        # Shutdown if active
        if self.executor is not None:
            # Log
            _LOGGER.info('Shutting down executor')

            # Shutdown
            self.executor.shutdown(wait=wait, cancel_futures=cancel_futures)

            # Set to None
            self._executor = None

            # Set to zero
            self._n_workers = 0

    def map(self, fn, /, *iterables, timeout=None, chunksize=1):
        """
        See Also
        --------
        concurrent.futures.ProcessPoolExecutor.map
        """

        # If not already started, start executor
        if self.executor is None:
            self.init()

        # Map
        return self.executor.map(fn,
                                 *iterables,
                                 timeout=timeout,
                                 chunksize=chunksize)

    def submit(self, fn, /, *args, **kwargs):
        """
        See Also
        --------
        concurrent.futures.ProcessPoolExecutor.submit
        """

        # If not already started, start executor
        if self.executor is None:
            self.init()

        # Map
        return self.executor.submit(fn, *args, **kwargs)

    def init(self,
             max_workers: int = None,
             *,
             ignore_init_error: bool = False,
             **kwargs):
        """
        Initialize loky executor.

        Parameters
        ----------
        max_workers: int, optional
            Set the maximum number of tasks that can be running in parallel in
            worker processes. The default number of workers can be changed
            using the env variable `HYBRIDQ_PARALLEL_MAX_WORKERS` (usually
            set to the number of available cpus).
        ignore_init_error: bool, optional
            If `False`, raise an error if executor is already initialized.
            Otherwise, quitely ignore the initialization.

        See Also
        --------
        concurrent.futures.ProcessPoolExecutor.map
        """
        from os import cpu_count, environ

        # Set mar_workers
        max_workers = int(
            environ.get(
                'HYBRIDQ_PARALLEL_MAX_WORKERS',
                cpu_count())) if max_workers is None else int(max_workers)

        if self.executor is None:
            from concurrent.futures import ProcessPoolExecutor

            # Log
            _LOGGER.info('Starting executor with %s workers', max_workers)

            # Set number of workers
            self._n_workers = max_workers

            # Start executor
            self._executor = ProcessPoolExecutor(max_workers=max_workers,
                                                 **kwargs)

        elif not ignore_init_error:
            raise SystemError("Parallel executor is already initialized.")


# Initialize executor
_EXECUTOR = Executor()


def get_n_workers():
    return _EXECUTOR.n_workers


def init(max_workers: int = None, *, ignore_init_error: bool = False, **kwargs):
    """
    Initialize `Executor`.

    Parameters
    ----------
    max_workers: int, optional
        Set the maximum number of tasks that can be running in parallel in
        worker processes. By default this is set to the number of CPUs on the
        host.
    ignore_init_error: bool, optional
        If `False`, raise an error if executor is already initialized.
        Otherwise, quitely ignore the initialization.

    See Also
    --------
    Executor.init
    """
    # Initialize executor
    _EXECUTOR.init(max_workers=max_workers,
                   ignore_init_error=ignore_init_error,
                   **kwargs)


def shutdown(**kwargs):
    """
    Shutdown `Executor`.

    See Also
    --------
    Executor.shutdown
    """
    # Shutdown executor
    _EXECUTOR.shutdown(**kwargs)


def restart(max_workers: int = None,
            *,
            ignore_init_error: bool = False,
            wait: bool = True,
            cancel_futures: bool = False,
            **kwargs):
    """
    Restart `Executor`.

    Parameters
    ----------
    max_workers: int, optional
        Set the maximum number of tasks that can be running in parallel in
        worker processes. By default this is set to the number of CPUs on the
        host.
    ignore_init_error: bool, optional
        If `False`, raise an error if executor is already initialized.
        Otherwise, quitely ignore the initialization.
    wait: bool, optional
        If `True` then shutdown will not return until all running futures have
        finished executing and the resources used by the executor have been
        reclaimed.
    cancel_futures: bool, optional
        If `True` then shutdown will cancel all pending futures. Futures that are
        completed or running will not be cancelled.

    Other Parameters
    ----------------
    kwargs
        Parameters to send to `Executor.init`.

    See Also
    --------
    Executor.init, Executor.shutdown
    """
    # Shutdown executor
    _EXECUTOR.shutdown(wait=wait, cancel_futures=cancel_futures)

    # Start again
    _EXECUTOR.init(max_workers=max_workers,
                   ignore_init_error=ignore_init_error,
                   **kwargs)


def _map(fn: callable,
         /,
         *iterables,
         verbose: bool = False,
         pickler: str = None,
         **kwargs):
    from tqdm.auto import tqdm

    # Initialize
    try:
        verbose = dict(verbose)
        if not 'disable' in verbose:
            verbose['disable'] = False
    except:
        verbose = dict(disable=not verbose)

    # Get total
    if not verbose['disable']:

        def _len(x):
            try:
                return len(x)
            except:
                return None

        try:
            verbose['total'] = min(
                *filter(lambda x: x is not None, map(_len, iterables)))
        except:
            pass

    # Run map
    for x in tqdm(
            _EXECUTOR.map(Function(fn, pickler=pickler), *iterables, **kwargs),
            **verbose):
        yield x


def map(fn: callable,
        /,
        *iterables,
        verbose: bool = False,
        pickler: str = None,
        **kwargs):
    """
    Returns an iterator equivalent to `map(fn, iter)`.

    Parameters
    ----------
    fn: callable,
        A callable that will take as many arguments as there are passed
        iterables.
    timeout: int, optional
        The maximum number of seconds to wait. If None, then there is no limit
        on the wait time.
    chunksize: int, optional
        If greater than one, the iterables will be chopped into chunks of size
        chunksize and submitted to the process pool.  If set to one, the items
        in the list will be sent one at a time.
    pickler: str, optional
        Use `pickler` as module to pickle functions.
    verbose: bool, optional
        Show progressbar.

    Returns
    -------
    iter
        An iterator equivalent to: `map(func, *iterables)` but the calls may be
        evaluated out-of-order.

    Raises
    ------
    TimeoutError:
        If the entire result iterator could not be generated before the given
        timeout.
    Exception:
        If fn(*args) raises for any values.
    """
    return _map(fn, *iterables, verbose=verbose, pickler=pickler, **kwargs)


def starmap(fn, *iterables, **kwargs):
    """
    Like `map()` except that the elements of the iterable are expected to be
    iterables that are unpacked as arguments.

    Hence an iterable of `[(1,2), (3, 4)]` results in `[func(1,2), func(3,4)]`.

    See Also
    --------
    hybridq_parallel.map
    """
    return _map(lambda x: fn(*x), *iterables, **kwargs)
