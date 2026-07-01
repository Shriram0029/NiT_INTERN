import re
import spacy
import logging

logger = logging.getLogger(__name__)

class EntityExtractor:
    def __init__(self, protected_entities):
        self.protected_entities = protected_entities
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("Spacy model 'en_core_web_sm' not found.")
            self.nlp = None

    def extract(self, text):
        entities = []
        
        if "OTP" in self.protected_entities:
            otp_matches = list(re.finditer(r'\b\d{6}\b', text))
            for m in reversed(otp_matches):
                entities.append({"type": "OTP", "value": m.group(), "start": m.start(), "end": m.end()})
                text = text[:m.start()] + "[OTP]" + text[m.end():]
                
        if "CARD" in self.protected_entities:
            card_matches = list(re.finditer(r'\b(?:\d{4}[-\s]?){3}\d{4}\b', text))
            for m in reversed(card_matches):
                entities.append({"type": "CARD", "value": m.group(), "start": m.start(), "end": m.end()})
                text = text[:m.start()] + "[CARD]" + text[m.end():]
                
        if self.nlp is not None:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in self.protected_entities:
                    entities.append({"type": ent.label_, "value": ent.text, "start": ent.start_char, "end": ent.end_char})
                    
        return text, entities
