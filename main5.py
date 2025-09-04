from transformers import pipeline

# Load the NLLB model for translation
translator = pipeline("translation", model="facebook/nllb-200-distilled-600M", src_lang="eng_Latn", tgt_lang="npi_Deva",device='cpu')
# translator = pipeline("translation", model="tencent/Hunyuan-MT-7B", src_lang="eng_Latn", tgt_lang="npi_Deva",device='cuda')

def transliterate(text):
    """
    Transliterate English text to Nepali Devanagari using NLLB model.
    """
    result = translator(text)
    return result[0]['translation_text']

if __name__ == "__main__":
    # user_input = input("Enter English text to transliterate to Nepali Devanagari: ")
    user_input2="The government’s decision to ban Facebook and other major social media platforms that failed to register with the Ministry of Communication and Information Technology has triggered widespread debate. While officials argue the move was necessary to enforce regulation following a Supreme Court order, critics warn of serious implications for communication, livelihoods, and freedom of expression."
    output = transliterate(user_input2)
    print("Transliterated text:", output)
