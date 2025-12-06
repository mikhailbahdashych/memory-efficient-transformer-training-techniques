"""
Tokenizer for Polish text using BPE (Byte-Pair Encoding) or GPT-2 pre-trained tokenizer.
"""

import json
from pathlib import Path
from typing import List, Optional, Union
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence


class PolishTokenizer:
    """
    Custom tokenizer for Polish text using BPE.
    """

    def __init__(self, vocab_size: int = 10000):
        """
        Initialize tokenizer.

        Args:
            vocab_size: Size of the vocabulary
        """
        self.vocab_size = vocab_size
        self.tokenizer: Optional[Tokenizer] = None

        # Special tokens
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.bos_token = "[BOS]"
        self.eos_token = "[EOS]"
        self.special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]

    def train(self, texts: List[str]) -> None:
        """
        Train tokenizer on provided texts.

        Args:
            texts: List of text strings to train on
        """
        # Create a BPE tokenizer
        self.tokenizer = Tokenizer(BPE(unk_token=self.unk_token))

        # Set up normalizer (lowercase and strip accents for Polish)
        self.tokenizer.normalizer = Sequence([NFD(), Lowercase()])

        # Set up pre-tokenizer (split on whitespace)
        self.tokenizer.pre_tokenizer = Whitespace()

        # Create trainer
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens,
            min_frequency=2,
            show_progress=True,
        )

        # Train the tokenizer
        print(f"Training tokenizer on {len(texts)} texts...")
        self.tokenizer.train_from_iterator(texts, trainer=trainer)

        # Enable padding
        self.tokenizer.enable_padding(pad_token=self.pad_token)

        # Enable truncation
        self.tokenizer.enable_truncation(max_length=512)

        print(f"Tokenizer trained! Vocabulary size: {self.tokenizer.get_vocab_size()}")

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text
            add_special_tokens: Whether to add BOS/EOS tokens

        Returns:
            List of token IDs
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained yet!")

        encoding = self.tokenizer.encode(text)
        ids = encoding.ids

        if add_special_tokens:
            bos_id = self.tokenizer.token_to_id(self.bos_token)
            eos_id = self.tokenizer.token_to_id(self.eos_token)
            ids = [bos_id] + ids + [eos_id]

        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.

        Args:
            ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained yet!")

        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def save(self, path: Path) -> None:
        """
        Save tokenizer to file.

        Args:
            path: Path to save tokenizer
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained yet!")

        path = Path(path)
        self.tokenizer.save(str(path))
        print(f"Tokenizer saved to {path}")

        # Save metadata
        metadata = {
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
        }
        metadata_path = path.parent / f"{path.stem}_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def load(self, path: Path) -> None:
        """
        Load tokenizer from file.

        Args:
            path: Path to load tokenizer from
        """
        path = Path(path)
        self.tokenizer = Tokenizer.from_file(str(path))
        print(f"Tokenizer loaded from {path}")

        # Load metadata
        metadata_path = path.parent / f"{path.stem}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                self.vocab_size = metadata["vocab_size"]
                self.special_tokens = metadata["special_tokens"]

    @property
    def pad_token_id(self) -> int:
        """Get PAD token ID."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained yet!")
        return self.tokenizer.token_to_id(self.pad_token)

    @property
    def unk_token_id(self) -> int:
        """Get UNK token ID."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained yet!")
        return self.tokenizer.token_to_id(self.unk_token)

    @property
    def bos_token_id(self) -> int:
        """Get BOS token ID."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained yet!")
        return self.tokenizer.token_to_id(self.bos_token)

    @property
    def eos_token_id(self) -> int:
        """Get EOS token ID."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained yet!")
        return self.tokenizer.token_to_id(self.eos_token)


class GPT2Tokenizer:
    """
    Wrapper for pre-trained GPT-2 BPE tokenizer.

    Features:
    - Uses official GPT-2 tokenizer (pre-trained, no training needed!)
    - Compatible interface with PolishTokenizer
    - Save/load functionality
    """

    def __init__(self, vocab_size: int = 50257):
        """
        Initialize GPT-2 tokenizer.

        Args:
            vocab_size: Not used (GPT-2 has fixed vocab of 50257), kept for interface compatibility
        """
        try:
            from transformers import GPT2TokenizerFast
        except ImportError:
            raise ImportError(
                "GPT-2 tokenizer requires the 'transformers' library. "
                "Install it with: uv add transformers"
            )

        # Load pre-trained GPT-2 tokenizer
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        self.vocab_size = len(self.tokenizer)  # 50257 for GPT-2

        # Map to our special token IDs for compatibility
        # GPT-2 doesn't have all these, so we'll use its tokens
        self.pad_token = self.tokenizer.eos_token  # GPT-2 doesn't have PAD, use EOS
        self.unk_token = self.tokenizer.unk_token if self.tokenizer.unk_token else "<|endoftext|>"
        self.bos_token = self.tokenizer.bos_token if self.tokenizer.bos_token else "<|endoftext|>"
        self.eos_token = self.tokenizer.eos_token  # <|endoftext|>

        self.special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]

    def train(self, texts: List[str]) -> None:
        """
        No training needed - GPT-2 tokenizer is pre-trained!

        Args:
            texts: Ignored (kept for interface compatibility)
        """
        print(f"Using pre-trained GPT-2 BPE tokenizer")
        print(f"  Vocabulary size: {self.vocab_size}")
        print(f"  Note: GPT-2 tokenizer is pre-trained, no training needed!")
        print(f"  Special tokens:")
        print(f"    PAD: {self.pad_token_id} ('{self.pad_token}')")
        print(f"    UNK: {self.unk_token_id} ('{self.unk_token}')")
        print(f"    BOS: {self.bos_token_id} ('{self.bos_token}')")
        print(f"    EOS: {self.eos_token_id} ('{self.eos_token}')")

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text
            add_special_tokens: Whether to add BOS/EOS tokens

        Returns:
            List of token IDs
        """
        # GPT-2 tokenizer doesn't add BOS by default, we'll add manually if requested
        ids = self.tokenizer.encode(text, add_special_tokens=False)

        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]

        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.

        Args:
            ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text
        """
        if skip_special_tokens:
            # Filter out our special tokens
            special_ids = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
            ids = [token_id for token_id in ids if token_id not in special_ids]

        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def save(self, path: Path) -> None:
        """
        Save tokenizer metadata to file.

        Args:
            path: Path to save tokenizer JSON
        """
        path = Path(path)

        # We don't need to save the tokenizer itself (it's pre-trained)
        # Just save metadata
        data = {
            "type": "gpt2",
            "vocab_size": self.vocab_size,
            "pretrained_model": "gpt2",
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"Tokenizer metadata saved to {path}")

    def load(self, path: Path) -> None:
        """
        Load tokenizer from file.

        Args:
            path: Path to tokenizer JSON file
        """
        try:
            from transformers import GPT2TokenizerFast
        except ImportError:
            raise ImportError(
                "GPT-2 tokenizer requires the 'transformers' library. "
                "Install it with: uv add transformers"
            )

        path = Path(path)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Reload pre-trained GPT-2 tokenizer
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        self.vocab_size = data["vocab_size"]

        # Load special tokens
        self.pad_token = data["pad_token"]
        self.unk_token = data["unk_token"]
        self.bos_token = data["bos_token"]
        self.eos_token = data["eos_token"]

        print(f"Tokenizer loaded from {path}")

    @property
    def pad_token_id(self) -> int:
        """Get PAD token ID."""
        return self.tokenizer.eos_token_id

    @property
    def unk_token_id(self) -> int:
        """Get UNK token ID."""
        return self.tokenizer.unk_token_id if self.tokenizer.unk_token_id else self.tokenizer.eos_token_id

    @property
    def bos_token_id(self) -> int:
        """Get BOS token ID."""
        return self.tokenizer.bos_token_id if self.tokenizer.bos_token_id else self.tokenizer.eos_token_id

    @property
    def eos_token_id(self) -> int:
        """Get EOS token ID."""
        return self.tokenizer.eos_token_id


# Factory functions

def create_tokenizer(tokenizer_type: str = "bpe", vocab_size: int = 10000) -> Union[PolishTokenizer, GPT2Tokenizer]:
    """
    Create a tokenizer of the specified type.

    Args:
        tokenizer_type: Type of tokenizer ('bpe' or 'gpt2')
        vocab_size: Vocabulary size (ignored for gpt2 - it has fixed vocab of 50257)

    Returns:
        Tokenizer instance

    Raises:
        ValueError: If tokenizer type is unknown
    """
    tokenizer_type = tokenizer_type.lower()

    if tokenizer_type == "bpe":
        return PolishTokenizer(vocab_size=vocab_size)
    elif tokenizer_type == "gpt2":
        return GPT2Tokenizer(vocab_size=vocab_size)  # vocab_size ignored, uses 50257
    else:
        raise ValueError(
            f"Unknown tokenizer type: {tokenizer_type}. "
            f"Choose from: 'bpe', 'gpt2'"
        )


def load_tokenizer(tokenizer_path: Path, tokenizer_type: str = None) -> Union[PolishTokenizer, GPT2Tokenizer]:
    """
    Load a tokenizer from file.

    Args:
        tokenizer_path: Path to tokenizer file
        tokenizer_type: Type of tokenizer (optional, auto-detected from file if not provided)

    Returns:
        Loaded tokenizer instance

    Raises:
        ValueError: If tokenizer type cannot be determined
    """
    # Try to auto-detect tokenizer type from file
    if tokenizer_type is None:
        try:
            with open(tokenizer_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                tokenizer_type = data.get('type', None)

            if tokenizer_type is None:
                # Check for tokenizer-specific fields
                if 'pretrained_model' in data and data['pretrained_model'] == 'gpt2':
                    tokenizer_type = 'gpt2'
                else:
                    # Default to BPE (PolishTokenizer) for backward compatibility
                    tokenizer_type = 'bpe'
        except Exception:
            # Default to BPE if can't determine
            tokenizer_type = 'bpe'

    # Create appropriate tokenizer and load
    tokenizer_type = tokenizer_type.lower()
    if tokenizer_type == "gpt2":
        tokenizer = GPT2Tokenizer()
        tokenizer.load(tokenizer_path)
        return tokenizer
    elif tokenizer_type == "bpe":
        tokenizer = PolishTokenizer()
        tokenizer.load(tokenizer_path)
        return tokenizer
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")


def get_tokenizer_name(tokenizer_type: str) -> str:
    """
    Get a human-readable name for the tokenizer type.

    Args:
        tokenizer_type: Type of tokenizer

    Returns:
        Human-readable name
    """
    names = {
        "bpe": "Custom BPE Tokenizer",
        "gpt2": "GPT-2 BPE (Pre-trained)",
    }
    return names.get(tokenizer_type.lower(), tokenizer_type)


def get_available_tokenizers() -> list:
    """
    Get list of available tokenizer types.

    Returns:
        List of tokenizer type strings
    """
    return ["bpe", "gpt2"]


if __name__ == "__main__":
    # Test tokenizer
    sample_texts = [
        "To jest przykładowy tekst po polsku.",
        "Witaj świecie! Jak się masz?",
        "Uczenie maszynowe jest fascynujące.",
    ]

    print("=" * 80)
    print("Testing BPE Tokenizer")
    print("=" * 80)

    tokenizer = PolishTokenizer(vocab_size=1000)
    tokenizer.train(sample_texts)

    # Test encoding/decoding
    text = "Witaj świecie!"
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    print(f"Original: {text}")
    print(f"Encoded: {ids}")
    print(f"Decoded: {decoded}")

    print("\n" + "=" * 80)
    print("Testing GPT-2 Tokenizer")
    print("=" * 80)

    try:
        gpt2_tokenizer = create_tokenizer("gpt2")
        gpt2_tokenizer.train(sample_texts)  # Does nothing for pre-trained

        text = "Witaj świecie!"
        ids = gpt2_tokenizer.encode(text)
        decoded = gpt2_tokenizer.decode(ids)

        print(f"Original: {text}")
        print(f"Encoded: {ids}")
        print(f"Decoded: {decoded}")
    except ImportError as e:
        print(f"GPT-2 tokenizer not available: {e}")
