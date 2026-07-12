"""Export-format versioning.

The export JSON is a versioned contract: every artifact produced by
``extract_export_data``/``SurrogateExport.to_dict`` is stamped with
``EXPORT_SCHEMA_VERSION``, and loaders check compatibility before trusting
the payload shape. This is what lets a client and server built against
different versions of this library still register/diff/score models: as
long as a loader's version is >= the payload's stamped version, it knows
how to read that (older) shape.
"""

EXPORT_SCHEMA_VERSION = 1


class IncompatibleExportVersionError(ValueError):
    """Raised when an export payload is newer than this library understands."""


def check_compatible(payload_version: int) -> bool:
    """Return True if this library can read a payload stamped with ``payload_version``."""
    return payload_version <= EXPORT_SCHEMA_VERSION
