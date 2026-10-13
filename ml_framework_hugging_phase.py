from transformers import pipeline
sentiment = pipeline("sentiment-analysis")

result = sentiment("I really enjoyed the picnic.")
print(result)

result = sentiment("The laboratory computers are very slow.")
print(result)

result = sentiment("Is the AI always correct?")
print(result)

result = sentiment("Are dogs better than cats?")
print(result)

result = sentiment("Attending classes regularly is good")
print(result)

result = sentiment("Honesty is the best policy.")
print(result)

result = sentiment("is studying regularly for a set of hours good?")
print(result)
