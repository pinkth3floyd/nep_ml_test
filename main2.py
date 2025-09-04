from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained('Sakonii/distilbert-base-nepali')
model = AutoModelForMaskedLM.from_pretrained('Sakonii/distilbert-base-nepali')

# prepare input
text = "चाहिएको text यता राख्नु होला।"
encoded_input = tokenizer(text, return_tensors='pt')

# forward pass
output = model(**encoded_input)

print(output)


decode = tokenizer.decode(output.logits[0].argmax(axis=-1))
print(decode)