import logging
from logging.handlers import RotatingFileHandler

import pytest

import logging_config


@pytest.fixture
def restore_safetylens_logger():
    logger = logging.getLogger("safetylens")
    original_handlers = logger.handlers[:]
    original_level = logger.level
    logger.handlers = []
    try:
        yield logger
    finally:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()
        logger.handlers = original_handlers
        logger.setLevel(original_level)


def test_logging_retention_is_bounded_and_invalid_env_falls_back(
    monkeypatch,
    tmp_path,
    restore_safetylens_logger,
):
    monkeypatch.setattr(logging_config, "LOGS_DIR", tmp_path)
    monkeypatch.setenv("SAFETYLENS_LOG_MAX_BYTES", "not-an-integer")
    monkeypatch.setenv("SAFETYLENS_LOG_BACKUP_COUNT", "also-invalid")

    logging_config.setup_logging()

    files = [
        handler
        for handler in restore_safetylens_logger.handlers
        if isinstance(handler, RotatingFileHandler)
    ]
    assert len(files) == 2
    assert {handler.maxBytes for handler in files} == {
        logging_config.DEFAULT_LOG_MAX_BYTES
    }
    assert {handler.backupCount for handler in files} == {
        logging_config.DEFAULT_LOG_BACKUP_COUNT
    }


def test_log_level_override_applies_to_console_and_files(
    monkeypatch,
    tmp_path,
    restore_safetylens_logger,
):
    monkeypatch.setattr(logging_config, "LOGS_DIR", tmp_path)
    monkeypatch.setenv("SAFETYLENS_LOG_LEVEL", "ERROR")

    logging_config.setup_logging()

    assert restore_safetylens_logger.handlers
    assert all(
        handler.level == logging.ERROR
        for handler in restore_safetylens_logger.handlers
    )


def test_reinitializing_logging_closes_replaced_handlers(
    monkeypatch,
    tmp_path,
    restore_safetylens_logger,
):
    monkeypatch.setattr(logging_config, "LOGS_DIR", tmp_path)
    previous = logging.StreamHandler()
    restore_safetylens_logger.addHandler(previous)

    logging_config.setup_logging()

    assert previous not in restore_safetylens_logger.handlers
    assert previous._closed is True
