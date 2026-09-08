"""Serialize access to the Fortran metric/flux tables between Python callers.

Each call still uses OpenMP internally. For independent concurrent scenes use
separate processes. The lock is reentrant for composed public API calls.
"""

from functools import wraps
from inspect import signature
from threading import RLock

CORE_LOCK = RLock()


def metric_context(func):
    first_parameter = next(iter(signature(func).parameters))

    @wraps(func)
    def wrapped(*args, **kwargs):
        if not args and first_parameter not in kwargs:
            return func(*args, **kwargs)  # preserve the normal missing-argument error
        spacetime = args[0] if args else kwargs[first_parameter]
        st = getattr(getattr(spacetime, "physical", None), "spacetime", spacetime)
        with CORE_LOCK:
            if hasattr(st, "_activate"):
                st._activate()
            elif getattr(st, "mid", None) not in (1, 2):
                raise ValueError(
                    "unregistered metric; use CustomMetric or TabulatedMetric"
                )
            return func(*args, **kwargs)

    return wrapped


def metadata(obj):
    import dataclasses
    import json

    if hasattr(obj, "metadata"):
        data = obj.metadata()
    else:
        data = {"type": type(obj).__name__, **dataclasses.asdict(obj)}
    return json.loads(json.dumps(data))
