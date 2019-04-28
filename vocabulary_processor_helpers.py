import re

from tensorflow.contrib import learn

_MATCHER_PIPELINE = [
    (re.compile(r'\.'), ' period '),
    (re.compile(r"’"), "'"),
    (re.compile(r"can't"), "can not"),
    (re.compile(r"wasn't"), "was not"),
    (re.compile(r"ca n't"), "can not"),
    (re.compile(r" n't "), " not "),
    (re.compile(r"( \% )"),  ' percentage '),
    (re.compile(r'(\# ?[X\d]+)'), 'check'),
    (re.compile(r'(\{?\$[X\d]+(?:\.\d+)?\}?)'), 'dollar'),
    (re.compile(r'([\dX]+/[\dX]+/[\dX]+)'), 'date'),
    (re.compile(r'([X]+)20\d\d'), 'date'),
    (re.compile(r'\W+',), ' '),
    (re.compile(r'\s{2,}'), ' '),
    (re.compile(r'([X]+) bank'), 'bank'),
]

_TOKENIZER_RE = re.compile(r"[^\s]+", re.UNICODE)

def build_vocabulary_processor(file, max_doc_length, min_freq):
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_doc_length, min_freq, 
                                                              tokenizer_fn=_tokenizer)
    vocab_processor.fit(text_document_iter(file))
    vocabulary_size = len(vocab_processor.vocabulary_)
    print("Model built with vocabulary of size %s" % (vocabulary_size))
    return vocab_processor

def text_document_iter(file):
    with open(file, 'r') as f:
        for row in f:
            yield row.split('\t')[1]
            
def _text_normalizer(s):
    s = s.replace('\\n', '\n').replace('\\r', '').replace('\\', '')
    for match_repl in _MATCHER_PIPELINE:
        s = match_repl[0].sub(match_repl[1], s)
    s = s.lower().strip()
    return s

def _tokenizer(iterator):
    for text in iterator:
        text = _text_normalizer(text)
        yield _TOKENIZER_RE.findall(text)