import re
from src.entity_extractor import EntityExtractor

class Preprocessor:
    def __init__(self, config):
        self.extractor = EntityExtractor(config.get('protected_entities', []))

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def process(self, complaint_text):
        clean = self.clean_text(complaint_text)
        processed_text, entities = self.extractor.extract(clean)
        return {
            "original": complaint_text,
            "processed_text": processed_text,
            "entities": entities
        }
