from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
tokenizer = AutoTokenizer.from_pretrained("Shushant/nepaliBERT")
model = AutoModelForMaskedLM.from_pretrained("Shushant/nepaliBERT")



fill_mask = pipeline( "fill-mask", model=model, tokenizer=tokenizer, device='cpu' )
print(fill_mask(f"तिमीलाई कस्तो {tokenizer.mask_token}."))
print(fill_mask(f"नेपालको राजधानी {tokenizer.mask_token} हो।"))   