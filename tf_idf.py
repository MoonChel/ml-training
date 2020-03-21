# TF-IDF
# https://www.youtube.com/watch?v=RPMYV-eb6lI

# naive
# term frequency * inverse document frequency
# means that if token appears often in multiple documents it will get lower score
# formula = frequency_in_current_document * log(count_of_all_documents / count_of_documents_where_word_appears)

import re
import math
from collections import defaultdict
from pprint import pprint

corpus = [
    "call me ishmael. some years ago—never mind how long precisely—having",
    "little or no money in my purse, and nothing particular to interest me",
    "on shore, i thought i would sail about a little and see the watery part",
    "of the world. it is a way i have of driving off the spleen and",
    "regulating the circulation. whenever i find myself growing grim about",
]
corpus_len = len(corpus)

regex = re.compile("[^a-zA-Z]")

tf_idf = defaultdict(float)

for line in corpus:
    # remove punctuation
    line = regex.sub(" ", line)

    words = line.split()
    word_appearing_count = defaultdict(int)

    for word in words:
        word_appearing_count[word] += 1

    for word, count in word_appearing_count.items():
        document_count = 1
        for line_2 in corpus:
            # split by whitespace
            if word in line_2.split():
                document_count += 1

        # +1 is add one smoothing
        # same applies for document_count = 1
        # more here https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting
        tf_idf[word] = count * math.log((corpus_len + 1) / document_count)

sorted_tf_idf = sorted(tf_idf.items(), key=lambda word_count: word_count[1])

pprint(sorted_tf_idf)
# result
# [
#     ("and", 0.22314355131420976),
#     ("i", 0.22314355131420976),
#     ("the", 0.22314355131420976),
#     ("me", 0.5108256237659907),
#     ("little", 0.5108256237659907),
#     ("about", 0.5108256237659907),
#     ("a", 0.5108256237659907),
#     ("call", 0.9162907318741551),
#     ("some", 0.9162907318741551),
#     ("years", 0.9162907318741551),
#     ("mind", 0.9162907318741551),
#     ("how", 0.9162907318741551),
#     ("long", 0.9162907318741551),
#     ("or", 0.9162907318741551),
#     ("no", 0.9162907318741551),
#     ("money", 0.9162907318741551),
#     ("in", 0.9162907318741551),
#     ("my", 0.9162907318741551),
#     ("nothing", 0.9162907318741551),
#     ("particular", 0.9162907318741551),
#     ("to", 0.9162907318741551),
#     ("interest", 0.9162907318741551),
#     ("on", 0.9162907318741551),
#     ("thought", 0.9162907318741551),
#     ("would", 0.9162907318741551),
#     ("sail", 0.9162907318741551),
#     ("see", 0.9162907318741551),
#     ("watery", 0.9162907318741551),
#     ("part", 0.9162907318741551),
#     ("it", 0.9162907318741551),
#     ("is", 0.9162907318741551),
#     ("way", 0.9162907318741551),
#     ("have", 0.9162907318741551),
#     ("driving", 0.9162907318741551),
#     ("off", 0.9162907318741551),
#     ("spleen", 0.9162907318741551),
#     ("regulating", 0.9162907318741551),
#     ("whenever", 0.9162907318741551),
#     ("find", 0.9162907318741551),
#     ("myself", 0.9162907318741551),
#     ("growing", 0.9162907318741551),
#     ("grim", 0.9162907318741551),
#     ("ishmael", 1.6094379124341003),
#     ("ago", 1.6094379124341003),
#     ("never", 1.6094379124341003),
#     ("precisely", 1.6094379124341003),
#     ("having", 1.6094379124341003),
#     ("purse", 1.6094379124341003),
#     ("shore", 1.6094379124341003),
#     ("world", 1.6094379124341003),
#     ("circulation", 1.6094379124341003),
#     ("of", 1.8325814637483102),
# ]
