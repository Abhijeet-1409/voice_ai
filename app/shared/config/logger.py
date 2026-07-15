from functools import cache
from contextvars import ContextVar
from logging import Logger, StreamHandler, Filter, Formatter ,getLogger

from shared.config.settings import get_app_settings, AppBaseSettings


# ── Context variable for stream_sid ─────────────────────────────────────────
stream_sid_var: ContextVar[str] = ContextVar("stream_sid", default="-")


class StreamSidFilter(Filter):
    """Custom logging filter to add stream_sid to log records"""
    
    def filter(self, record):
        """
        Inject the current stream_sid into the log record.

        Reads the stream_sid from the `stream_sid_var` context variable
        (defaults to "-" if not set in the current context) and attaches
        it to the record as `record.stream_sid`, making it available to
        any formatter that references %(stream_sid)s.

        :param record: the LogRecord being processed
        :return: True always, so the record is never filtered out
        """

        record.stream_sid = stream_sid_var.get()
        return True
    

@cache
def get_base_logger() -> Logger:
    """
    Get the root logger for the application, configuring it on first call.

    On the first call, this builds the root logger ("intelics_bot"),
    applies the configured log level, attaches a console handler with
    the configured formatter and the StreamSidFilter, and returns it.
    Because this function is cached, all subsequent calls return the
    same already-configured Logger instance without repeating setup.

    :return: the configured root Logger instance
    """

    # Base settings
    base_settings: AppBaseSettings = get_app_settings()

    # Root logger
    root_logger = getLogger("intelics_bot")
    root_logger.setLevel(base_settings.LOG_LEVEL)

    # Root formatter 
    formatter = Formatter(base_settings.LOG_FORMAT, base_settings.DATA_FORMAT)

    # Console handler
    console_handler = StreamHandler()
    console_handler.setLevel(base_settings.LOG_LEVEL)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(StreamSidFilter())

    # Adding console handler to root logger
    root_logger.addHandler(console_handler)
    
    return root_logger


@cache
def get_logger(name: str) -> Logger:
    """
    Get a named child logger of the application's root logger.

    The returned logger inherits the root logger's level and console
    handler via propagation, so messages logged through it are printed
    to console automatically. Callers may attach additional handlers
    (e.g. a FileHandler) directly to the returned logger if needed.

    Calls with the same `name` return the same cached Logger instance,
    so any handlers added by one caller persist for all other callers
    using that same name.

    :param name: name for the child logger (e.g. a service or module name)
    :return: a Logger instance scoped to the given name
    """

    root_logger = get_base_logger()
    named_logger = root_logger.getChild(name)
    return named_logger
