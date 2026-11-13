words = [
    # Местоимения (10)
    "I", "You", "He", "She", "We",
    "They", "My", "Your", "Our", "Their",

    # Глаголы (10)
    "love", "hate", "like", "dislike", "enjoy",
    "prefer", "want", "need", "know", "think",

    # Прилагательные (10)
    "good", "bad", "great", "terrible", "amazing",
    "boring", "interesting", "beautiful", "awful", "wonderful",

    # Наречия (10)
    "very", "really", "quite", "extremely", "always",
    "never", "sometimes", "usually", "too", "not",

    # Программирование (10)
    "Python", "Java", "JavaScript", "TypeScript", "Go",
    "programming", "coding", "software", "website", "application",

    # Животные (10)
    "cats", "dogs", "birds", "fish", "horses",
    "rabbits", "lions", "tigers", "elephants", "monkeys",

    # Развлечения (10)
    "music", "movies", "games", "books", "football",
    "basketball", "chess", "art", "travel", "sports",

    # Еда (10)
    "pizza", "burger", "pasta", "sushi", "chocolate",
    "coffee", "tea", "apples", "bananas", "bread",

    # Дополнительные слова (10)
    "and", "but", "because", "with", "without",
    "is", "are", "am", "do", "does",

    # Существительные (10)
    "friend", "family", "school", "work", "computer",
    "phone", "car", "house", "city", "world"
]

word_to_idx = {word: idx for idx, word in enumerate(words)}


subjects = [
    "I", "You", "He", "She", "We",
    "They", "My", "Your", "Our", "Their"
]

objects = [
    "Python", "Java", "music", "cats", "dogs",
    "games", "books", "pizza", "movies", "sports"
]

positive = ["love", "like"]
negative = ["hate", "dislike"]

sentences = []
labels = []

for subject in subjects:
    for obj in objects:
        sentences.append(f"{subject} {positive[0]} {obj}")
        labels.append(1)

        sentences.append(f"{subject} {negative[0]} {obj}")
        labels.append(0)
print(labels)
print(sentences)