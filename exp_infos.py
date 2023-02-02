# MODELS INFOS
LM_MODELS={
    "gpt2":{
        'family':'gpt2',
        'n_p':117e6,
    },
    "gpt2-medium":
    {
        'family':'gpt2',
        'n_p':345e6,
    },
    "gpt2-large":
    {
        'family':'gpt2',
        'n_p':774e6,
    },
    "gpt2-xl":
    {
        'family':'gpt2',
        'n_p':1.5e9,
    },
    "facebook/opt-350m":
    {
        'family':'opt',
        'n_p':350e6,
    },
    "facebook/opt-1.3b": # GPU 1, BS 32
    {
        'family':'opt',
        'n_p':1.3e9,
    },
    "facebook/opt-6.7b": # GPU 1, BS 32
    {
        'family':'opt',
        'n_p':6.7e9,
    },
        "facebook/opt-13b": # GPU 2, BS 32
    {
        'family':'opt',
        'n_p':13e9,
    },
    "facebook/opt-30b": # GPU 4, BS 32
    {
        'family':'opt',
        'n_p':30e9,
    },
    "facebook/opt-66b":
    {
        'family':'opt',
        'n_p':66e9,
    },
    "facebook/opt-iml-max-30b": # GPU 4, BS 32
    {
        'family':'opt-iml',
        'n_p':30e9,
    },
    "facebook/opt-iml-max-1.3b": # GPU 1, BS 32
    {
        'family':'opt-iml',
        'n_p':1.3e9,
    },
    "facebook/galactica-6.7b": # GPU 1, BS 32
    {
        'family':'galactica',
        'n_p':6.7e9,
    },
    "facebook/galactica-30b": # GPU 4, BS 32
    {
        'family':'galactica',
        'n_p':30e9,
    },
}

# corpus-size in GB
LM_FAMILIES={
    'gpt2':
    {
        'type':'causal',
        'finetuned':'',
        'tokenizer':'bpe',
        'pretrain-corpus':[
            'webtext'
        ],
        'pretrain-corpus-size':40,
    },
    'opt':
    {
        'type':'causal',
        'finetuned':'',
        'tokenizer':'bpe',
        'pretrain-corpus':[
            'bookcorpus',
            'cc-stories',
            'the-pile',
            'pushshift-reddit',
            'cc-news-v2'
        ],
        'pretrain-corpus-size':800,
    },
    'opt-iml':
    {
        'type':'causal',
        'finetuned':'auto-instruct',
        'tokenizer':'bpe',
        'pretrain-corpus':[
            'bookcorpus',
            'cc-stories',
            'the-pile',
            'pushshift-reddit',
            'cc-news-v2'
        ],
        'pretrain-corpus-size':800,
        'finetune-corpus':[
            'super-naturalinstructions',
            'promptsource',
            'crossfit',
            'flan',
            'exmix',
            't5',
            'unifiedskg',
            'reasoning'
        ]
    },
    'galactica':
    {
        'type':'causal',
        'finetuned':'',
        'tokenizer':'bpe',
        'pretrain-corpus':[ # corpus > 100 million tokens
            'papers', # 88B tokens (e.g. arxiv)
            'code', # academic github
            'knowledge bases', # 2B tokens (e.g. pubchem coumpoumd) 
            'wikipedia', # 5B tokens
            'stackexchange', # 1B tokens
            'libretext', # 185M tokens
            'wikibooks', # 110M tokens
            'filtered cc', # 1.1B tokens. Scientific and Academic
            'prompts', # 358M tokens
        ],
        'pretrain-corpus-size':-1, # 106B tokens
    },
}

PROMPTS={
    'LAMA_relations':{
        'type':'human',
        'path':'prompts/LAMA_relations.jsonl'
    }
}