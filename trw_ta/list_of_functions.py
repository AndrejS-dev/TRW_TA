import inspect
import trw_ta

def register_outputs(*columns):
    def decorator(func):
        func._outputs = columns
        return func
    return decorator

def function_list():
    print("List of available functions:\n")
    for name, func in inspect.getmembers(trw_ta, inspect.isfunction):
        if func.__module__.startswith('trw_ta'):
            sig = inspect.signature(func)
            outputs = getattr(func, '_outputs', ['...'])
            output_list = f"df[{outputs}]" if len(outputs) > 1 else f"df[['{outputs[0]}']]"
            print(f"{name}{sig} -> {output_list} ({len(outputs)} column{'s' if len(outputs) != 1 else ''})")
