import sys
import inspect
import logging
from typing import Literal
from . import PACKAGE_NAME
from .singleton_metaclass import SingletonMeta


class ColorFormatter(logging.Formatter):
    RED = "\033[31m"
    YELLOW = "\033[33m"
    BOLD_RED = "\033[1;31m"
    RESET = "\033[0m"

    def format(self, record):
        if record.levelno == logging.CRITICAL:
            color = self.BOLD_RED
        elif record.levelno == logging.ERROR:
            color = self.RED
        elif record.levelno == logging.WARNING:
            color = self.YELLOW
        else:
            color = self.RESET

        formatted = super().format(record)

        # If we have additional stack lines stored in the record, append them now,
        # ensuring they're placed AFTER the [filename:lineno] part.
        stack_lines = getattr(record, 'stack_lines', '')
        if stack_lines:
            formatted += "\n" + stack_lines

        formatted = f"{color}{formatted}{self.RESET}"

        return formatted

class Logger(logging.Logger, metaclass=SingletonMeta):
    _LINES_TO_LOG = {
        'ERROR': 3,
        'CRITICAL': 3,
    }

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        return instance

    def __init__(self, level: int | str = logging.DEBUG) -> None:
        if getattr(self, "_initialized", False):
            return
        super().__init__(PACKAGE_NAME, level)
        self._initialize_logger()

    def _initialize_logger(self):
        self._initialized = True
        self.handlers.clear()
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        formatter = ColorFormatter('%(asctime)s - %(levelname)s: %(message)s')
        console_handler.setFormatter(formatter)
        self.addHandler(console_handler)

    def _log_last_lines(self, lines: int) -> tuple[str, str]:
        stack = inspect.stack()[3:]  # 0 = _log_last_lines, 1 = __log, 2 = debug/info/warning/error/critical, 3 = original caller
        lines = min(lines, len(stack))
        last_x_frames = stack[:lines]
        caller = last_x_frames.pop(0)
        stack_lines = [f"\t{frame.filename} ({frame.function}:{frame.lineno})" for frame in last_x_frames]
        if stack_lines: stack_lines = ['Stack:'] + stack_lines
        return f"({caller.filename.split('/')[-1]}:{caller.function}:{caller.lineno})", "\n".join(stack_lines)

    def set_level(self, level: str | int) -> None:
        self.setLevel(level)

    def set_default_log_lines(self, lines: int, level: str | int | None = None) -> None:
        if level:
            if isinstance(level, int):
                level = logging.getLevelName(level)
            levels = [level.upper()]
        else:
            levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        for l in levels:
            self._LINES_TO_LOG[l] = lines

    def __log(self, level: Literal['debug', 'info', 'warning', 'error', 'critical'], msg: str, *args, stack_lines: int | None = None, **kwargs) -> None:
        lines = stack_lines if stack_lines is not None else self._LINES_TO_LOG.get(level.upper(), 0)
        kwargs.pop('stack_lines', None)
        caller, stack_info = self._log_last_lines(lines+1)
        msg = f'{msg} {caller}'
        extra = kwargs.pop('extra', {})
        extra['stack_lines'] = stack_info
        kwargs['extra'] = extra
        getattr(super(), level)(msg, *args, **kwargs)

    def debug(self, msg: str, *args, stack_lines: int | None = None, **kwargs) -> None:
        self.__log('debug', msg, *args, stack_lines=stack_lines, **kwargs)

    def info(self, msg: str, *args, stack_lines: int | None = None, **kwargs) -> None:
        self.__log('info', msg, *args, stack_lines=stack_lines, **kwargs)

    def warning(self, msg: str, *args, stack_lines: int | None = None, **kwargs) -> None:
        self.__log('warning', msg, *args, stack_lines=stack_lines, **kwargs)

    def error(self, msg: str, *args, stack_lines: int | None = None, **kwargs) -> None:
        self.__log('error', msg, *args, stack_lines=stack_lines, **kwargs)

    def critical(self, msg: str, *args, stack_lines: int | None = None, **kwargs) -> None:
        self.__log('critical', msg, *args, stack_lines=stack_lines, **kwargs)
