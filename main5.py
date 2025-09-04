from transformers import pipeline

# Load the NLLB model for translation
translator = pipeline("translation", model="facebook/nllb-200-distilled-600M", src_lang="eng_Latn", tgt_lang="npi_Deva",device='cpu')

def transliterate(text):
    """
    Transliterate English text to Nepali Devanagari using NLLB model.
    """
    result = translator(text)
    return result[0]['translation_text']

if __name__ == "__main__":
    user_input = input("Enter English text to transliterate to Nepali Devanagari: ")
    output = transliterate(user_input)
    print("Transliterated text:", output)
