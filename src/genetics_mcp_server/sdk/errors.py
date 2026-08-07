"""Errors raised by the genetics SDK."""


class GeneticsError(RuntimeError):
    """A data request failed.

    The tool layer returns `{"success": False, "error": ...}` because a model reads the
    dict. A script author does not check a flag after every call, so the SDK raises instead
    — a failure that is ignored would otherwise show up as an empty DataFrame.
    """


class GeneticsUsageError(GeneticsError):
    """The arguments given to an SDK function do not select exactly one query shape."""
