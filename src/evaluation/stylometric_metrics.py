import numpy as np
from collections import Counter
import spacy

POS_TAGS_ORDERED = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]
FUNCTION_WORD_POS = {"DET", "PRON", "ADP", "AUX", "CCONJ", "SCONJ", "PART"}

ACTIVE_FEATURE_GROUPS = [
    "avg_sentence_length",
    "avg_word_length",
    "root_ttr",
    "pos_unigram_ratios",
    "function_word_ratio",
]

DEFAULT_MAX_POS_BIGRAMS = 50

ONIX_STOPWORDS = [
    "a","about","above","across","after","again","against","all","almost","alone",
    "along","already","also","although","always","among","an","and","another","any",
    "anybody","anyone","anything","anywhere","are","area","areas","around","as","ask",
    "asked","asking","asks","at","away","back","backed","backing","backs","be","became",
    "because","become","becomes","been","before","began","behind","being","beings","best",
    "better","between","big","both","but","by","came","can","cannot","case","cases",
    "certain","certainly","clear","clearly","come","could","did","differ","different",
    "differently","do","does","done","down","downed","downing","downs","during","each",
    "early","either","end","ended","ending","ends","enough","even","evenly","ever","every",
    "everybody","everyone","everything","everywhere","face","faces","fact","facts","far",
    "felt","few","find","finds","first","for","four","from","full","fully","further",
    "furthered","furthering","furthers","gave","general","generally","get","gets","give",
    "given","gives","go","going","good","goods","got","great","greater","greatest","group",
    "grouped","grouping","groups","had","has","have","having","he","her","here","herself",
    "high","higher","highest","him","himself","his","how","however","if","important",
    "in","interest","interested","interesting","interests","into","is","it","its","itself",
    "just","keep","keeps","kind","knew","know","known","knows","large","largely","last",
    "later","latest","least","less","let","lets","like","likely","long","longer","longest",
    "made","make","making","man","many","may","me","member","members","men","might","more",
    "most","mostly","mr","mrs","much","must","my","myself","necessary","need","needed",
    "needing","needs","never","new","newer","newest","next","no","nobody","non","noone",
    "not","nothing","now","nowhere","number","numbers","of","off","often","old","older",
    "oldest","on","once","one","only","open","opened","opening","opens","or","order",
    "ordered","ordering","orders","other","others","our","out","over","part","parted",
    "parting","parts","per","perhaps","place","places","point","pointed","pointing","points",
    "possible","present","presented","presenting","presents","problem","problems","put","puts",
    "quite","rather","really","right","room","rooms","said","same","saw","say","says",
    "second","seconds","see","seem","seemed","seeming","seems","sees","several","shall","she",
    "should","show","showed","showing","shows","side","sides","since","small","smaller",
    "smallest","so","some","somebody","someone","something","somewhere","state","states",
    "still","such","sure","take","taken","than","that","the","their","them","then","there",
    "therefore","these","they","thing","things","think","thinks","this","those","though",
    "thought","thoughts","three","through","thus","to","today","together","too","took",
    "toward","turn","turned","turning","turns","two","under","until","up","upon","us","use",
    "used","uses","very","want","wanted","wanting","wants","was","way","ways","we","well",
    "wells","went","were","what","when","where","whether","which","while","who","whole",
    "whose","why","will","with","within","without","work","worked","working","works","would",
    "year","years","yet","you","young","younger","youngest","your","yours",
]
ONIX_STOPWORDS_SET = set(ONIX_STOPWORDS)

def _has_feature(feature_name):
    return feature_name in ACTIVE_FEATURE_GROUPS


def load_spacy_model(model_name="en_core_web_sm", require_gpu=True):
    """Load a spaCy pipeline on GPU when requested.

    If require_gpu=True and GPU is unavailable, raises a RuntimeError so callers
    do not silently run vector extraction on CPU.
    """
    if require_gpu:
        try:
            spacy.require_gpu()
        except Exception as exc:
            raise RuntimeError(
                "GPU is required for stylometric vector extraction but is not available."
            ) from exc
    else:
        spacy.prefer_gpu()

    return spacy.load(model_name)


def fit_feature_vocabulary(
    texts,
    nlp_model,
    max_pos_bigrams=DEFAULT_MAX_POS_BIGRAMS,
):
    """Fit a shared POS bigram vocabulary from a corpus.

    The returned vocabulary should be reused across corpora so extracted vectors
    keep the same dimensionality.
    """
    bigram_counts = Counter()

    for doc in nlp_model.pipe(texts, batch_size=256, disable=["ner"]):
        tokens_alpha = [token for token in doc if token.is_alpha]
        if _has_feature("pos_bigram_ratios"):
            pos_tags = [token.pos_ for token in tokens_alpha]
            bigram_counts.update(zip(pos_tags, pos_tags[1:]))

    return {
        "pos_bigrams": [
            bigram for bigram, _ in bigram_counts.most_common(max_pos_bigrams)
        ] if _has_feature("pos_bigram_ratios") else [],
    }


def get_feature_keys(vocab=None):
    """Get the ordered list of feature keys for vectorization."""
    vocab = vocab or {"pos_bigrams": []}
    feature_keys = []

    for base_feature in ["avg_sentence_length", "avg_word_length", "root_ttr"]:
        if _has_feature(base_feature):
            feature_keys.append(base_feature)

    if _has_feature("function_word_ratio"):
        feature_keys.append("function_word_ratio")

    if _has_feature("pos_unigram_ratios"):
        feature_keys.extend([f"POS_{p}" for p in POS_TAGS_ORDERED])

    if _has_feature("pos_bigram_ratios"):
        feature_keys.extend(
            [f"POS2_{left}_{right}" for left, right in vocab.get("pos_bigrams", [])]
        )

    return feature_keys


def extract_features_batch(texts, nlp_model, vocab=None, use_cfg=False):
    """Process a list of texts efficiently using nlp.pipe."""
    vocab = vocab or {"pos_bigrams": []}
    feature_keys = get_feature_keys(vocab)
    pos_bigram_vocab = list(vocab.get("pos_bigrams", []))
    all_vectors = []

    # 2. Process in batches (disable NER for speed)
    for doc in nlp_model.pipe(texts, batch_size=256, disable=["ner"]):
        sentences = list(doc.sents)
        tokens = [t for t in doc if t.is_alpha]
        num_tokens = len(tokens)

        # Basic Stats
        sent_lengths = [len([t for t in s if t.is_alpha]) for s in sentences]
        avg_sent_len = float(np.mean(sent_lengths)) if sent_lengths else 0.0
        
        word_lengths = [len(t.text) for t in tokens]
        avg_word_len = float(np.mean(word_lengths)) if word_lengths else 0.0

        # POS & Function Words
        pos_counts = Counter(t.pos_ for t in tokens)
        total_pos = sum(pos_counts.values())
        func_count = sum(1 for t in tokens if t.lower_ in ONIX_STOPWORDS_SET)
        func_ratio = (func_count / num_tokens) if num_tokens > 0 else 0.0

        # Lexical Diversity
        unique_lemmas = set(t.lemma_.lower() for t in tokens)
        root_ttr = (len(unique_lemmas) / np.sqrt(num_tokens)) if num_tokens > 0 else 0.0

        # 3. Build Feature Dictionary for this Doc
        current_feats = {}
        if _has_feature("avg_sentence_length"):
            current_feats["avg_sentence_length"] = avg_sent_len
        if _has_feature("avg_word_length"):
            current_feats["avg_word_length"] = avg_word_len
        if _has_feature("root_ttr"):
            current_feats["root_ttr"] = root_ttr
        if _has_feature("function_word_ratio"):
            current_feats["function_word_ratio"] = func_ratio
        
        # 4. add POS distributions
        if _has_feature("pos_unigram_ratios"):
            for pos in POS_TAGS_ORDERED:
                current_feats[f"POS_{pos}"] = (pos_counts[pos] / total_pos) if total_pos > 0 else 0.0

        if _has_feature("pos_bigram_ratios"):
            pos_tags = [t.pos_ for t in tokens]
            pos_bigrams = list(zip(pos_tags, pos_tags[1:]))
            total_bigrams = len(pos_bigrams)
            bigram_counts = Counter(pos_bigrams)
            for left, right in pos_bigram_vocab:
                feature_name = f"POS2_{left}_{right}"
                if total_bigrams > 0:
                    current_feats[feature_name] = bigram_counts[(left, right)] / total_bigrams
                else:
                    current_feats[feature_name] = 0.0

        # 4. Vectorize (Ensuring 0.0 for missing features)
        vector = [current_feats.get(k, 0.0) for k in feature_keys]
        all_vectors.append(vector)

    return np.array(all_vectors)
