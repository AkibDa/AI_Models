# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
#
# tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
# model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
#
# text_to_translate = "Life is like a box of chocolates"
# model_inputs = tokenizer(text_to_translate, return_tensors="pt")
#
# # translate to French
# gen_tokens = model.generate(**model_inputs, forced_bos_token_id=tokenizer.get_lang_id("fr"))
# print(tokenizer.batch_decode(gen_tokens, skip_special_tokens=True))

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

article = "UN Chief says there is no military solution in Syria"
inputs = tokenizer(article, return_tensors="pt")

translated_tokens = model.generate(
    **inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids("fra_Latn"), max_length=30
)
print(tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0])

#https://huggingface.co/docs/transformers/main/en/model_doc/nllb