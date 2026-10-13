
# ============================================================
# HUGGING FACE SENTIMENT ANALYSIS
# ============================================================

from transformers import pipeline


def main():
    # 1. Define the model
    model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

    try:
        print("Loading Hugging Face sentiment analysis model...")

        sentiment_analyzer = pipeline(
            task="sentiment-analysis",
            model=model_name
        )

        # 2. Input sentences
        texts = [
            "I really enjoyed the picnic.",
            "The laboratory computers are very slow.",
            "Is the AI always correct?"
            "Studying regularly is beneficial"
            "Pune has horrible traffic"
            "Does shopping makes me feel happy?"
            "Is going out on vacation good?"
            "Gyming is good for health"

            ]

        # 3. Validate inputs
        texts = [
            text.strip()
            for text in texts
            if isinstance(text, str) and text.strip()
        ]

        if not texts:
            print("Error: No valid text was provided.")
            return

        # 4. Perform sentiment analysis in a batch
        results = sentiment_analyzer(texts)

        # 5. Display results
        print("\n" + "=" * 60)
        print("HUGGING FACE SENTIMENT ANALYSIS RESULTS")
        print("=" * 60)

        for index, (text, result) in enumerate(
            zip(texts, results), start=1
        ):
            label = result["label"]
            score = result["score"]

            print(f"\nInput {index}: {text}")
            print(f"Sentiment: {label}")
            print(f"Confidence: {score:.4f} ({score * 100:.2f}%)")

    except ImportError as error:
        print("Required libraries or ML backend are missing.")
        print("Install transformers and PyTorch.")
        print("Details:", error)

    except OSError as error:
        print("The model could not be loaded.")
        print("Check your internet connection and model access.")
        print("Details:", error)

    except Exception as error:
        print("An unexpected error occurred:")
        print(f"{type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
