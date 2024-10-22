from transformers import pipeline

classifier = pipeline(
    task="text-classification",
    model="cffl/bert-base-styleclassification-subjective-neutral",
    return_all_scores=True,
)

def is_subjective(text):
    result = classifier(text)
    if classifier.model.name_or_path == 'cffl/bert-base-styleclassification-subjective-neutral':
        assert result[0][0]['label'] == 'SUBJECTIVE'
        return 1 if result[0][0]['score'] > 0.5 else 0
    elif classifier.model.name_or_path == 'GroNLP/mdebertav3-subjectivity-multilingual':
        assert result[0][1]['label'] == 'LABEL_1'
        return 1 if result[0][1]['score'] > 0.5 else 0
    elif classifier.model.name_or_path == 'Re0x10/subjectivity-detection-for-ChatGPT-sentiment':
        assert result[0][1]['label'] == 'subjective'
        return 1 if result[0][1]['score'] > 0.5 else 0