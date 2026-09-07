from livekit.plugins.cartesia import TTS

from speech.output_language import OutputLanguage


class CartesiaOutputLanguage(OutputLanguage):
    """
    Cartesia-specific implementation for managing text-to-speech output language state.

    Wraps a LiveKit Cartesia TTS plugin instance to dynamically update and retrieve 
    its synthesis language options during an active session using an internally cached state.
    """

    SUPPORTED_LANGUAGES: dict[str, str] = {
        "en": "English",
        "fr": "French",
        "de": "German",
        "es": "Spanish",
        "pt": "Portuguese",
        "zh": "Chinese",
        "ja": "Japanese",
        "hi": "Hindi",
        "it": "Italian",
        "ko": "Korean",
        "nl": "Dutch",
        "pl": "Polish",
        "ru": "Russian",
        "sv": "Swedish",
        "tr": "Turkish",
        "tl": "Tagalog",
        "bg": "Bulgarian",
        "ro": "Romanian",
        "ar": "Arabic",
        "cs": "Czech",
        "el": "Greek",
        "fi": "Finnish",
        "hr": "Croatian",
        "ms": "Malay",
        "sk": "Slovak",
        "da": "Danish",
        "ta": "Tamil",
        "uk": "Ukrainian",
        "hu": "Hungarian",
        "no": "Norwegian",
        "vi": "Vietnamese",
        "bn": "Bengali",
        "th": "Thai",
        "he": "Hebrew",
        "ka": "Georgian",
        "id": "Indonesian",
        "te": "Telugu",
        "gu": "Gujarati",
        "kn": "Kannada",
        "ml": "Malayalam",
        "mr": "Marathi",
        "pa": "Punjabi",
    }

    def __init__(self, language: str, tts: TTS):
        """
        Initializes the CartesiaOutputLanguage controller.

        Args:
            language (str): The initial ISO language code for the TTS plugin.
            tts (TTS): The LiveKit Cartesia TTS plugin instance to manage.

        Raises:
            ValueError: If the initial language code is not supported by Cartesia.
        """
        self._tts = tts
        self._supported_languages = self.SUPPORTED_LANGUAGES
        
        if language not in self._supported_languages:
            raise ValueError(
                f"Unsupported initial language code '{language}'. "
                f"Supported options: {list(self._supported_languages.keys())}"
            )
            
        self._language = language

    @property
    def supported_languages(self) -> dict[str, str]:
        """
        Retrieves the map of language codes to display names supported by Cartesia.

        Returns:
            dict[str, str]: Dictionary mapping language codes (e.g., 'en') to language names (e.g., 'English').
        """
        return self._supported_languages.copy()

    def set_language(self, language: str) -> None:
        """
        Updates the target synthesis language on the Cartesia TTS instance and internal cache.

        Args:
            language (str): The ISO language code to set for upcoming TTS output.

        Raises:
            ValueError: If the provided language code is not present in supported languages.
        """
        if language not in self._supported_languages:
            raise ValueError(
                f"Unsupported language code '{language}'. "
                f"Supported options: {list(self._supported_languages.keys())}"
            )
        self._language = language
        self._tts.update_options(language=language)

    def get_language(self) -> str | None:
        """
        Retrieves the currently tracked language code.

        Returns:
            str | None: The active ISO language code (e.g., 'en', 'es'), or None if unset.
        """
        return self._language