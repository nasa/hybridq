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

__all__ = ['map', 'starmap', 'get_n_workers', 'init', 'shutdown', 'restart']

# create logger
_LOGGER = logging.getLogger(__name__)
_LOGGER.setLevel(logging.INFO)
_LOGGER_CH = logging.StreamHandler()
_LOGGER_CH.setLevel(logging.DEBUG)
_LOGGER_CH.setFormatter(
    logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s'))
_LOGGER.addHandler(_LOGGER_CH)

# Initialize executor
_HYBRIDQ_PARALLEL_EXECUTOR = None


def get_n_workers():
    global _HYBRIDQ_PARALLEL_EXECUTOR
    return _HYBRIDQ_PARALLEL_EXECUTOR._max_workers


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
    loky.get_reusable_executor
    """
    from loky import get_reusable_executor
    global _HYBRIDQ_PARALLEL_EXECUTOR

    # If _HYBRIDQ_PARALLEL_EXECUTOR already initialize, raise
    if _HYBRIDQ_PARALLEL_EXECUTOR is not None:
        if ignore_init_error:
            _LOGGER.warning("'Executor' already initialized")
        else:
            raise RuntimeError("'Executor' already initialized")

    # Initialize
    else:
        # Initialize executor
        _HYBRIDQ_PARALLEL_EXECUTOR = get_reusable_executor(
            max_workers=max_workers, **kwargs)

        # Log
        _LOGGER.info(f"Started 'Executor' with "
                     f"{_HYBRIDQ_PARALLEL_EXECUTOR._max_workers} workers")


def shutdown(**kwargs):
    """
    Shutdown `Executor`.

    See Also
    --------
    loky.reusable_executor._ReusablePoolExecutor.shutdown
    """
    global _HYBRIDQ_PARALLEL_EXECUTOR

    # Shutdown executor
    if _HYBRIDQ_PARALLEL_EXECUTOR is not None:
        _HYBRIDQ_PARALLEL_EXECUTOR.shutdown(**kwargs)
        _HYBRIDQ_PARALLEL_EXECUTOR = None

        # Log
        _LOGGER.info(f"Shut down 'Executor'")
    else:
        # Log
        _LOGGER.warning(f"No 'Executor' to shut down")


def restart(max_workers: int = None,
            *,
            ignore_init_error: bool = False,
            wait: bool = True,
            kill_workers: bool = False,
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
    kill_workers: bool, optional
        If `True` then shutdown will cancel all pending futures. Futures that
        are completed or running will not be cancelled.

    Other Parameters
    ----------------
    kwargs
        Parameters to send to `Executor.init`.

    See Also
    --------
    loky.reusable_executor._ReusablePoolExecutor.init
    loky.reusable_executor._ReusablePoolExecutor.shutdown
    """
    global _HYBRIDQ_PARALLEL_EXECUTOR

    # Shutdown executor
    shutdown(wait=wait, kill_workers=kill_workers)

    # Start again
    init(max_workers=max_workers, ignore_init_error=ignore_init_error, **kwargs)


def _map(fn: callable, /, *iterables, **kwargs):
    global _HYBRIDQ_PARALLEL_EXECUTOR

    # Autostart
    if _HYBRIDQ_PARALLEL_EXECUTOR is None:
        if kwargs.pop('autostart', False):
            init()
        else:
            raise RuntimeError("'Executor' not starter")

    # Run map
    for x in _HYBRIDQ_PARALLEL_EXECUTOR.map(fn, *iterables, **kwargs):
        yield x


def map(fn: callable, /, *iterables, autostart: bool = True, **kwargs):
    """
    Returns an iterator equivalent to `map(fn, iter)`.

    Parameters
    ----------
    fn: callable,
        A callable that will take as many arguments as there are passed
        iterables.
    autostart: bool, optional
        If `True`, start a new Executor with default parameters.
    timeout: int, optional
        The maximum number of seconds to wait. If None, then there is no limit
        on the wait time.
    chunksize: int, optional
        If greater than one, the iterables will be chopped into chunks of size
        chunksize and submitted to the process pool.  If set to one, the items
        in the list will be sent one at a time.

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

    See Also
    --------
    map
    """
    return _map(fn, *iterables, autostart=autostart, **kwargs)


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
