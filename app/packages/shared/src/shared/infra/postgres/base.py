from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    """
    Unified declarative base class for all SQLAlchemy ORM models.

    Inheriting from this class automatically registers subclasses with the
    SQLAlchemy registry, enabling declarative mapping of Python classes to
    database tables and managing database schema metadata.
    """

    pass