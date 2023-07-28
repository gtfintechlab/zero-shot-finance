from nltk import word_tokenize


def finer_ord_decode(label_word):
    label_word = label_word.lower()
    if "other" in label_word:
        return 0
    elif "person_b" in label_word:
        return 1
    elif "person_i" in label_word:
        return 2
    elif "location_b" in label_word:
        return 3
    elif "location_i" in label_word:
        return 4
    elif "organisation_b" in label_word:
        return 5
    elif "organisation_i" in label_word:
        return 6
    else:
        return -1


def fomc_communication_decode(x):
    try:
        list_words = word_tokenize(x)
        label_word = list_words[0].lower()
        if "dovish" in label_word:
            return 0
        elif "hawkish" in label_word:
            return 1
        elif "neutral" in label_word:
            return 2
        else:
            return -1
    except:
        return -1


def numclaim_detection_decode(x):
    list_words = word_tokenize(x)
    label_word = list_words[0].lower()
    if "outofclaim" in label_word:
        return 0
    elif "inclaim" in label_word:
        return 1
    else:
        return -1


def sentiment_analysis_decode(x):
    list_words = word_tokenize(x)
    label_word = list_words[0].lower()
    if "positive" in label_word:
        return 0
    elif "negative" in label_word:
        return 1
    elif "neutral" in label_word:
        return 2
    else:
        return -1
