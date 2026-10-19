import inspect
import trw_ta

def register_outputs(*columns):
    """
    Decorator that attaches expected output column names to an indicator function.

    Parameters
    ----------
    *columns : str
        One or more column names produced by the wrapped function.

    Returns
    -------
    callable
        Decorator that sets ``func._outputs`` to ``columns``.
    """
    def decorator(func):
        func._outputs = columns
        return func
    return decorator

def function_list():
    """
    Print every public ``trw_ta`` function with its signature and output columns.

    Intended for interactive discovery in notebooks or the REPL. Output is written
    to stdout.
    """
    print("List of available functions:\n")
    for name, func in inspect.getmembers(trw_ta, inspect.isfunction):
        if func.__module__.startswith('trw_ta'):
            sig = inspect.signature(func)
            outputs = getattr(func, '_outputs', ['...'])
            output_list = f"df[{outputs}]" if len(outputs) > 1 else f"df[['{outputs[0]}']]"
            print(f"{name}{sig} -> {output_list} ({len(outputs)} column{'s' if len(outputs) != 1 else ''})")
