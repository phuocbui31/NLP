from typing import List
import re

from src.core.interfaces import Tokenizer

class SimpleTokenizer(Tokenizer):
    def tokenize(self, text: str) -> List[str]:
        text = text.lower()
        
        tokens = re.findall(r'\b\w+\b|[.,!?]', text)
        return tokens
