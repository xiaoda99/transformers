from random import choice, choices, shuffle, sample, randint, random, seed
from dataclasses import dataclass
from pattern.en import lexeme
# from nltk.corpus import cmudict  # nltk.download('cmudict')
 
from transformers import AutoTokenizer, GPT2Tokenizer
from common_utils import Timer

cache_dir = '/nas/xd/.cache/torch/transformers/'
with Timer('In const.py: Loading tokenizer'): tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6B', local_files_only=True, cache_dir=cache_dir)

# from https://eslyes.com/namesdict/popular_names.htm
boys = [
    'James', 'David',  'Christopher',  'George',  'Ronald',
    'John', 'Richard',  'Daniel',  'Kenneth',  'Anthony',
    'Robert', 'Charles',  'Paul',  'Steven',  'Kevin',
    'Michael', 'Joseph',  'Mark',  'Edward',  'Jason',
    'William',  'Thomas',  'Donald',  'Brian',  'Jeff',]
girls = [
    'Mary','Jennifer', 'Lisa', 'Sandra', 'Michelle',
    'Patricia','Maria', 'Nancy', 'Donna', 'Laura',
    'Linda','Susan', 'Karen', 'Carol', 'Sarah',
    'Barbara','Margaret', 'Betty', 'Ruth', 'Kimberly',
    'Elizabeth', 'Dorothy', 'Helen', 'Sharon', 'Deborah',]

def all_persons(tokenizer): 
    if not hasattr(all_persons, 'boys'):
        # https://www.verywellfamily.com/top-1000-baby-boy-names-2757618
        # https://www.verywellfamily.com/top-1000-baby-girl-names-2757832
        boys = [l.strip() for l in open('boy_names_1000.txt').readlines()]
        girls = [l.strip() for l in open('girl_names_1000.txt').readlines()]

        girls = [name for name in girls if max(len(tokenizer.tokenize(name)), len(tokenizer.tokenize(' ' + name))) == 1]
        boys = [name for name in boys if max(len(tokenizer.tokenize(name)), len(tokenizer.tokenize(' ' + name))) == 1]
        boys = sample(boys, len(girls))
        all_persons.boys, all_persons.girls = boys, girls
    return all_persons.boys, all_persons.girls

def persons():
    boys, girls = all_persons(tokenizer)
    return boys + girls

def genders_of_persons(): return {'boy': boys, 'girl': girls}

verb_form =[
    ('sleep','slept'),
    ('go','went'),
    ('come','came'),
    ('leave','left'),
    ('talk','talked'),
    ('speak','spoke'),
    ('hear','heard'),
    ('listen','listened'),
    ('see','saw'),
    ('look','looked'),
    ('eat','ate'),
    ('drink','drank'),
    ('stand','stood'),
    ('sit','sat'),
    ('walk','walked'),
    ('run','ran'),
    ('swim','swam'),
    ('fly','flew'),
    ('sing','sang'),
    ('dance','danced'),
    ('fall','fell'),
    ('write','wrote'),
    ('draw','drew'),
    ('drive','drove'),
    ('ride','rode'),
    ('play','played'),
    ('forget','forgot'),
    ('know','knew'),
    ('read','read'),
    ('cut','cut'),
    ('hit','hit'),
    ('hurt','hurt'),
    # ('can','could'),
    # ('do','did'),
    # ('are','were'),
    # ('begin','began'),
    # ('take','took'),
    # ('have','had'),
    # ('try','tried'),
    # ('want','wanted'),
]

# or conjugate(verb='give',tense=PRESENT,number=SG)
# https://stackoverflow.com/questions/3753021/using-nltk-and-wordnet-how-do-i-convert-simple-tense-verb-into-its-present-pas
@dataclass
class Tenses:
    do: str = None
    does: str = None
    doing: str = None
    did: str = None
    done: str = None

def verb_tenses():
    if not hasattr(verb_tenses, '_verb_tenses'):
        verbs = [v for v, _ in verb_form]
        try: _verb_tenses = [lexeme(v) for v in verbs]
        except: _verb_tenses = [lexeme(v) for v in verbs]
        verb_tenses._verb_tenses = [Tenses(*(vt + [vt[0]] * (5 - len(vt)))) for vt in _verb_tenses]
    return verb_tenses._verb_tenses

def does2did():
    d = {vt.does: [vt.did] for vt in verb_tenses()}
    d['sings'] = ['sang']; d['leaves'] = ['left']
    return d
    
noun2adj = [  # The adjective form of x is y
    ('rain','rainy'),
    ('sun','sunny'),
    ('friend','friendly'),
    ('danger','dangerous'),
    ('difference','different'),
    ('sadness','sad'),
    ('progress','progressive'),
    ('success','successful'),
    ('wisdom','wise'),
    ('love','loving'),
    ('kindness','kind'),
    ('truth','true'),
    ('beauty','beautiful'),
    ('freedom','free'),
    ('courage','courageous'),
    ('silence','silent'),
]