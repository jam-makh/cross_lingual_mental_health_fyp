from predictor import Predictor


CONFIG_PATH = "config_exported.yaml"

TEST_SENTENCES = [
    "Je me sens triste et sans espoir ces derniers temps.",
    "Aujourd'hui je me sens bien et calme.",
]


def main():
    print("=" * 70)
    print("TESTING EXPORTED FRENCH LSTM + TF-IDF MODEL")
    print("=" * 70)

    predictor = Predictor(CONFIG_PATH)

    for text in TEST_SENTENCES:
        label, confidence = predictor.predict(text)

        print("\nText:")
        print(text)
        print("Prediction:", label)
        print("Confidence:", f"{confidence:.2%}")

    print("\n" + "=" * 70)
    print("If predictions and confidence appear, the French export works.")
    print("=" * 70)


if __name__ == "__main__":
    main()
