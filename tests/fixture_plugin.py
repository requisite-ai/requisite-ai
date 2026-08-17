"""A real, importable module used as an EntryPoint.load() target in
tests/test_plugins.py -- not a mock. `register_call_count` is reset by
each test that relies on it, since module state persists across the
whole test session once imported.
"""

register_call_count = 0


def register() -> None:
    global register_call_count
    register_call_count += 1
