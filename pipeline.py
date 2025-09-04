from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForMaskedLM, BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F


class NepaliGrammarCorrector:
    def __init__(self, ged_path, token_path, mlm_path):
        self.ged_tokenizer = BertTokenizer.from_pretrained(ged_path)
        self.ged_model = BertForSequenceClassification.from_pretrained(ged_path)
        self.ged_model.eval()

        self.tok_tokenizer = AutoTokenizer.from_pretrained(ged_path)
        self.token_model = AutoModelForTokenClassification.from_pretrained(token_path)
        self.token_model.eval()

        self.mlm_tokenizer = AutoTokenizer.from_pretrained(mlm_path)
        self.mlm_model = AutoModelForMaskedLM.from_pretrained(mlm_path)
        self.mlm_model.eval()

    def is_nepali_word(self, word):
        return (
            all('\u0900' <= ch <= '\u097F' for ch in word if ch.isalpha())
            and any(ch in word for ch in "अआइईउऊऋएऐओऔकखगघचछजझटठडढतथदधनपफबभमयरलवशषसह")
        )

    def is_sentence_incorrect(self, sentence):
        inputs = self.ged_tokenizer(sentence, return_tensors="pt", truncation=True, padding=True).to(self.ged_model.device)
        with torch.no_grad():
            logits = self.ged_model(**inputs).logits
            probs = F.softmax(logits, dim=1)
        return torch.argmax(probs).item() == 1

    def get_incorrect_token_positions(self, sentence):
        tokens = self.tok_tokenizer(sentence, return_tensors="pt", return_offsets_mapping=True, truncation=True)
        if 'offset_mapping' in tokens:
            del tokens['offset_mapping']
        word_ids = self.tok_tokenizer(sentence).word_ids()
        with torch.no_grad():
            logits = self.token_model(**tokens).logits
        preds = torch.argmax(logits, dim=-1).squeeze().tolist()
        incorrect_positions = [i for i, (p, w) in enumerate(zip(preds, word_ids)) if p == 1 and w is not None]
        return incorrect_positions, tokens

    def suggest_corrections(self, tokens, incorrect_positions, top_k=5):
        masked_input = tokens["input_ids"].clone()

        for i in incorrect_positions:
            masked_input[0, i] = self.mlm_tokenizer.mask_token_id

        with torch.no_grad():
            logits = self.mlm_model(input_ids=masked_input).logits

        suggestions = {}
        for i in incorrect_positions:
            top_ids = torch.topk(logits[0, i], k=top_k).indices
            words = self.mlm_tokenizer.convert_ids_to_tokens(top_ids.tolist())
            words = [w.replace("▁", "").replace("##", "") for w in words if self.is_nepali_word(w)]
            suggestions[i] = words[:top_k]
        return suggestions

    def apply_corrections(self, tokens, suggestions):
        input_ids = tokens["input_ids"].squeeze().tolist()
        corrected = input_ids[:]
        for i, options in suggestions.items():
            if options:
                corrected[i] = self.mlm_tokenizer.convert_tokens_to_ids(options[0])
        return self.mlm_tokenizer.decode(corrected, skip_special_tokens=True)

    def correct(self, sentence):
        if not self.is_sentence_incorrect(sentence):
            return {"status": "Correct", "original": sentence}

        incorrect_positions, tokens = self.get_incorrect_token_positions(sentence)
        all_tokens = self.mlm_tokenizer.convert_ids_to_tokens(tokens["input_ids"].squeeze())
        incorrect_tokens = [all_tokens[i] for i in incorrect_positions]
        suggestions = self.suggest_corrections(tokens, incorrect_positions)
        corrected = self.apply_corrections(tokens, suggestions)

        return {
            "status": "Incorrect",
            "original": sentence,
            "tokens": all_tokens,
            "incorrect_positions": incorrect_positions,
            "incorrect_tokens": incorrect_tokens,
            "suggestions": suggestions,
            "corrected_sentence": corrected
        }