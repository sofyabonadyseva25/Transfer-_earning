from transformers import pipeline
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="blanchefort/rubert-base-cased-sentiment"
)
texts = [
    "Мне очень понравился этот фильм!",
    "Мне не понравилась еда в ресторане.",
    "Обычный день, ничего особенного."
]
results = sentiment_analyzer(texts)
for text, result in zip(texts, results):
    print(f"Текст: {text}")
    print(f"Эмоция: {result['label']}, уверенность: {result['score']:.3f}\n")
