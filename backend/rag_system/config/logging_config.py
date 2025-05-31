# backend/rag_system/config/logging_config.py
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any

try:
    import structlog
    from structlog.types import FilteringBoundLogger
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False
    FilteringBoundLogger = Any  # Type hint fallback


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    enable_json: bool = False,
    enable_colors: bool = True,
) -> Optional[Any]:  # Returns FilteringBoundLogger if structlog available
    """
    Setup production-grade structured logging with both console and file output.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        enable_json: Enable JSON formatting for structured logs
        enable_colors: Enable colored console output
        
    Returns:
        Configured structlog logger if available, None otherwise
    """
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure standard library logging first
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    if enable_json:
        console_formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"name": "%(name)s", "message": "%(message)s"}'
        )
    else:
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    console_handler.setFormatter(console_formatter)
    handlers.append(console_handler)
    
    # File handler (with rotation)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(numeric_level)
        
        # Always use JSON for file logs in production
        file_formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"name": "%(name)s", "message": "%(message)s", '
            '"module": "%(module)s", "function": "%(funcName)s", "line": %(lineno)d}'
        )
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add new handlers
    for handler in handlers:
        root_logger.addHandler(handler)
    
    # Silence noisy third-party loggers
    silence_loggers = [
        'chromadb',
        'sentence_transformers',
        'transformers',
        'torch',
        'urllib3.connectionpool',
        'httpx',
        'httpcore',
        'uvicorn.access',  # Reduce uvicorn access log noise
    ]
    
    for logger_name in silence_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    # Set up structlog if available
    if STRUCTLOG_AVAILABLE:
        # Configure timestamper
        timestamper = structlog.processors.TimeStamper(fmt="ISO")
        
        # Shared processors
        shared_processors = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            timestamper,
        ]
        
        if enable_json:
            # JSON formatting for production
            shared_processors.extend([
                structlog.processors.dict_tracebacks,
                structlog.processors.JSONRenderer()
            ])
        else:
            # Human-readable formatting for development
            shared_processors.extend([
                structlog.processors.ExceptionRenderer(),
                structlog.dev.ConsoleRenderer(colors=enable_colors)
            ])

        # Configure structlog
        structlog.configure(
            processors=shared_processors,
            wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )
        
        # Create and return structlog logger
        logger = structlog.get_logger("rag_system")
        logger.info(
            "Logging configured",
            level=log_level,
            file_logging=bool(log_file),
            json_format=enable_json,
            colored_output=enable_colors,
            structlog_available=True
        )
        return logger
    else:
        # Fallback to standard logging
        logger = logging.getLogger("rag_system")
        logger.info(f"Logging configured (standard library): level={log_level}, file={bool(log_file)}")
        return None


def get_logger(name: str = "rag_system") -> Any:
    """Get a logger instance."""
    if STRUCTLOG_AVAILABLE:
        return structlog.get_logger(name)
    else:
        return logging.getLogger(name)


# Fallback functions for when structlog is not available
def log_performance(operation: str, duration_ms: float, **kwargs: Any) -> None:
    """Log performance metrics."""
    logger = get_logger("rag_system.performance")
    if hasattr(logger, 'info'):
        logger.info(f"Performance: {operation} took {duration_ms:.2f}ms", **kwargs)
    else:
        logger.info(f"Performance: {operation} took {duration_ms:.2f}ms")


def log_security_event(event_type: str, details: Dict[str, Any], severity: str = "INFO") -> None:
    """Log security events."""
    logger = get_logger("rag_system.security")
    log_method = getattr(logger, severity.lower(), logger.info)
    log_method(f"Security event: {event_type}", **details)
