from context import Meta

meta = Meta(overhead=True)

# we need to put complex stuff in here :(

@meta("make CountVectorizer to encode word data as a vector")
def make_vectorizer(docs, max_ngram):
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(stop_words='english', ngram_range=(1,max_ngram))
    vecs = cv.fit_transform(docs)
    return {"vector":vecs, "vectorizer":cv}

print(make_vectorizer.__meta_id__)
print(make_vectorizer.__meta_source__)

vecs = make_vectorizer(["hello there","yeah that is right"],1)
vecs = make_vectorizer(["hello there","yeah that is right"],1)
vecs = make_vectorizer(["hello there","yeah that is right"],1)
vecs = make_vectorizer(["hello there","yeah that is right"],1)

print(vecs["vectorizer"])

print(meta.overhead_call)
print(meta.overhead_load)
