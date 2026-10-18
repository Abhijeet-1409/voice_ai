class CalendarClientError(Exception):
    """Base exception for calendar client errors."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message