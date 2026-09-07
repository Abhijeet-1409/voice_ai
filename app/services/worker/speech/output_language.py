from abc import ABC, abstractmethod


class OutputLanguage(ABC):
    """
    Abstract base class for output language management in text-to-speech (TTS) systems.

    Defines the contract for controlling dynamic language switching during active 
    agent sessions. Subclasses must implement the concrete logic required to update 
    the target language on underlying synthesis or model configurations.
    """

    @abstractmethod
    def set_language(self, language: str) -> None:
        """
        Sets the target language code for subsequent text-to-speech outputs.

        Args:
            language (str): The language code or locale identifier (e.g., 'en', 'es', 'hi') 
                to be applied to the output voice or model synthesis settings.

        Raises:
            NotImplementedError: If a concrete subclass does not implement this method.
        """
        raise NotImplementedError

    @abstractmethod
    def get_language(self) -> str | None:
        """
        Retrieves the currently configured output language code.

        Returns:
            str | None: The current language code or locale identifier in use for TTS outputs, or None if not configured.

        Raises:
            NotImplementedError: If a concrete subclass does not implement this method.
        """
        raise NotImplementedError