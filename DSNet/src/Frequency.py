from collections import Counter

def FreqModel(sentences, word_num=10):
    sentence = ' '.join(sentences)

    word_list = sentence.split()
    counter = Counter(word_list)
    
    hashtags = []
    for keyword, _ in counter.most_common(word_num):
        hashtags.append(keyword)

    return hashtags