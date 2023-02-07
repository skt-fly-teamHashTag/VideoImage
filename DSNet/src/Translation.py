from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def TranslationModel(sentences):
    model_name = "QuoQA-NLP/KE-T5-En2Ko-Base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    translated = model.generate(**tokenizer(sentences, return_tensors="pt", padding=True))
    ko_sentences = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    
    return ko_sentences