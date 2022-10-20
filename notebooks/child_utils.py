import sys
import os
import json
import csv
from collections import defaultdict, OrderedDict, Counter
import string
from random import choice, choices, shuffle, sample, randint, random, seed
from dataclasses import dataclass
from typing import Callable, Any

from pattern.en import conjugate, lemma, lexeme, PRESENT, SG
import nltk
from nltk.corpus import cmudict  # nltk.download('cmudict')

from common_utils import join_lists

sys.path.insert(0, '/nas/xd/projects/PyFunctional')
from functional import seq


# from transformers import GPT2Tokenizer
# cache_dir = '/nas/xd/.cache/torch/transformers/'
# _tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir=cache_dir)

uppercase = list(string.ascii_uppercase)
lowercase = list(string.ascii_lowercase)
digits = list(string.digits[1:])
cardinals = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
ordinals = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth']
digit2cardinal = OrderedDict(zip(digits, cardinals))
digit2ordinal = OrderedDict(zip(digits, ordinals))

# uppercases = [l for l in string.ascii_uppercase if len(_tokenizer.tokenize('%s %s' % (l*2, l*2))) == 2]
# lowercases = [l for l in string.ascii_lowercase if len(_tokenizer.tokenize('%s %s' % (l.upper()*2, l.upper()*2))) == 2]
full_vocab = uppercase + digits

# polygons = ['triangle', 'quadrangle', 'pentagon', 'hexagon', 'heptagon', 'octagon', 'nonagon', 'decagon',]# 'undecagon', 'dodecagon']
times_of_day = ['dawn', 'morning', 'noon', 'afternoon', 'evening', 'night',]# 'midnight']
clock_of_day = [f"{i} o'clock" for i in range (1, 13)]
years = [f'{i}' for i in range(2010, 2020)]
days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
seasons = ['spring', 'summer', 'autumn', 'winter']
# ages_of_life = ['baby', 'child', 'teenager', 'young', 'adult', 'elder']
ages_of_life = ['baby', 'child', 'adolescent', 'adult']
times_of_history = ['ancient', 'medieval', 'modern', 'contemporary'] #'renaissance', 
# units_of_time = ['nanosecond', 'microsecond', 'millisecond', ][:0] + ['second', 'minute', 'hour', 'day', 'week', 'month', 'year', 'decade', 'century', 'millennium'] # first 3 multi-token
units_of_length = ['nanometer', 'micrometer', 'millimeter', 'meter', 'kilometer', 'mile']
units_of_mass = ['nanogram', 'microgram', 'milligram', 'gram', 'kilogram', 'ton']
# SI_prefixes_small = ['pico', 'nana', 'micro', 'milli', 'centi', 'deci']
# SI_prefixes_large = ['kilo', 'mega', 'giga', 'tera', 'peta', 'exa', 'zetta', 'yotta']

things = ['atom', 'molecule', 'cell', 'tissue', 'organ', 'system', 'person', 'community', 'city', 'state', 'country', 'continent', 'planet', 'star', 'galaxy', 'universe']
sizes = ['tiny', 'small', 'large', 'huge',]# 'medium', 'gigantic']
# degrees = ['bachelor', 'master', 'doctor', 'postdoc']
posets = [list(string.ascii_uppercase)[:14], list(string.ascii_lowercase)[:14], list(string.ascii_uppercase)[14:], list(string.ascii_lowercase)[14:], digits, cardinals, ordinals,
    times_of_day, days_of_week, months, seasons, ages_of_life, times_of_history, #units_of_time, 
    things, sizes]# units_of_length, units_of_mass, SI_prefixes_small, SI_prefixes_large]
closed_posets = [list(string.ascii_uppercase)[:7], list(string.ascii_lowercase)[:7],][:] + [digits, cardinals, #ordinals[:5], 
    days_of_week, months, ]#seasons, times_of_history, ages_of_life, sizes]
open_posets = [times_of_day, ages_of_life, times_of_history, units_of_length, units_of_mass, things, sizes, ]

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

# https://www.verywellfamily.com/top-1000-baby-boy-names-2757618
# https://www.verywellfamily.com/top-1000-baby-girl-names-2757832
boys = [l.strip() for l in open('boy_names_1000.txt').readlines()]
girls = [l.strip() for l in open('girl_names_1000.txt').readlines()]
persons = boys + girls



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

lxy = [  # The adjective form of x is y
    ('apple','Apple'),
    ('high','High'),
    ('kill','Kill'),
    ('local','Local'),
    ('sun','Sun'),
    ('human','Human'),
    ('photo','Photo'),
    ('success','Success'),
    ('wake','Wake'),
    ('love','Love'),
    ('wear','Wear'),
    ('truth','Truth'),
    ('order','Order'),
    ('freedom','Freedom'),
    ('watch','Watch'),
    ('special','Special'),
]

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

verbs = [v for v, _ in verb_form]
try: verb_tenses = [lexeme(v) for v in verbs]
except: verb_tenses = [lexeme(v) for v in verbs]
verb_tenses = [Tenses(*(vt + [vt[0]] * (5 - len(vt)))) for vt in verb_tenses]

antonyms = [
    ('big', 'small'),
    ('long', 'short'),
    ('thick', 'thin'),
    ('tall', 'short'),
    ('strong', 'weak'),
    ('high', 'low'),
    ('hard', 'soft'),
    ('fast', 'slow'),
    ('light', 'dark'),
    ('up', 'down'),
    ('left', 'right'),
    ('near', 'far'),
    ('inside', 'outside'),
    ('front', 'back'),
    ('push', 'pull'),
    ('open', 'close'),
    ('hot', 'cold'),
    ('happy', 'sad'),
    ('loud', 'quiet'),
    ('good', 'bad'),
    ('right', 'wrong'),
    ('rich', 'poor'),
    ('clever', 'stupid'),
    ('male', 'female'),
    ('man', 'woman'),
    ('true', 'false'),
]

'''gpt-3 prompt
Types of tools: hammer, screwdriver, saw, drill, wrench
Types of clothes: shirt, pants, dress, coat, shoes
Types of fruits: apple, grape, pear, banana, orage
Types of animals: dog, cat, horse, rabbit, pig'''
types_of_things = {
    'animal': ['chicken', 'duck', 'goose', 'dog', 'lion', 'cow', 'donkey', 'horse', 'sheep', 'goat', 'bear', 'tiger', 'cat', 
            'zebra', 'pig', 'giraffe', 'monkey', 'rabbit', 'elephant', 'wolf', 'lion', 'deer', 'fox', 'gorilla', 'kangaroo'],
    'insect': ['bee', 'ant', 'fly', 'mosquito', 'wasp', 'butterfly', 'beetle', 'spider'],
    'flower': ['rose', 'tulip', 'lily', 'daisy', 'sunflower'],
    'fruit': ['apple', 'banana', 'pear', 'grape', 'cherry', 'orange', 'peach', 'plum', 'lemon', 'mango', 'blackberry',
            'blueberry', 'strawberry', 'durian', 'papaya', 'watermelon', 'pineapple', 'kiwi', 'apricot', 'lime'],
    'vehicle': ['car', 'bus', 'tractor', 'airplane', 'ship', 'bicycle', 'truck', 'train', 'motorbike', 'helicopter', 'carriage', 
                'subway', 'taxi', 'van', 'boat'],  # transportation
    'weapon': ['gun', 'rifle', 'sword', 'pistol', 'dagger', 'bomb', 'grenade', 'cannon'],
    'furniture': ['sofa', 'couch', 'desk', 'chair', 'table', 'bed', 'bookshelf'],# 'closet', 'wardrobe'],
    'tool': ['hammer', 'spanner', 'awl', 'scissors', 'axe', 'saw', 'shovel', 'screwdriver', 'wrench', 'drill', 'pliers'],
    'clothing': ['shirt', 'pants', 'dress', 'coat', 'socks', 'hat', 'tie', 'jacket', 'skirt', 'trousers', 'jeans'], #, 'shoes'
    'appliance': ['microwave', 'fridge', 'washer', 'dryer', 'washing machine'],  #, 'oven'
    # 'plant': ['tree', 'grass', 'bush', 'weed', 'vine'],
    # 'electronics': ['computer', 'laptop', 'iPad', 'phone', 'smartphone', 'television', 'camera', 'printer'],
    # 'utensil': ['spoon', 'fork', 'knife', 'plate', 'cup', 'bowl', 'pot'],
    # 'stationery': ['pen', 'pencil', 'paper', 'eraser', 'notebook', 'book', 'ruler', 'ink', 'stapler', 'rubber'],
}
# A list of words with their types:
# big small -> size
# blue red -> color
# 2 3 -> number
# cat dog -> animal
# German France -> country
# A B -> letter
# Febrary September -> month
# spring autumn -> season
# 2008 2017 -> year
# young old -> age
# ofen rarely occasionally -> frequency
# warm hot cold cool -> temperature
# Sunday Monday -> day
# fast slow -> speed

capabilities = [ # A x can y.
    ('knife', 'cut'),
    ('calculator', 'calculate'), # or compute
    ('phone', 'call'),
    ('printer', 'print'),
    ('pen', 'write'),
    ('saw', 'saw'),  # cut
    ('oven', 'bake'),
    ('pot', 'cook'), # boil
    ('gun', 'shoot'),
    ('dagger', 'stab'),
    # ('pan', 'fry'),
    ('brush', 'paint'),
    ('shovel', 'dig'),
    ('hammer', 'hit'),
    ('axe', 'chop'),
    ('drill', 'bore'),  # or drill
    ('lamp', 'light'),
    ('fan', 'blow'),
    ('washing machine', 'wash'),  # or washer
    ('opener ', 'open'),  # or washer
    ('dryer', 'dry'),
    ('lighter', 'light'),
    
    ('TV', 'watch'),  # show
    ('car', 'drive'),

    ('plane', 'fly'),
    ('bicycle', 'ride'),
    ('glider', 'glide'),
    ('skateboard', 'skate'),
    ('swing', 'swing'),
    ('piano ', 'play'),
    ('violin  ', 'play'),
    # ('book', 'read'), # teach
]

adj2very = [
    ('good', 'excellent'),
    ('bad', 'horrible'),
    ('fat', 'obese'),
    ('thin', 'skinny'),
    ('clean', 'spotless'),
    ('dirty', 'filthy'),
    ('big', 'huge'),
    ('small', 'tiny'),
    ('clever', 'intelligent'),
    ('stupid', 'idiotic'),
    ('easy', 'effortless'),
    ('hard', 'gruelling'),
    ('boring', 'tedious'),
    ('interesting', 'fascinating'),
    ('cold', 'freezing'),
    ('hot', 'boiling'),
    ('sad', 'miserable'),
    ('happy', 'ecstatic'),
]

en2fr = [
    ('apple', 'pomme'),
    ('cat', 'chat'),
    ('banana', 'banane'),
    # ('watermelon', 'pastèque'),
    ('morning', 'matin'),
    ('butter', 'beurre'),
    ('cheese', 'fromage'),
    ('dog', 'chien'),
    ('sugar', 'sucre'),
    ('coffee', 'café'),
    ('tea', 'thé'),
    ('juice', 'jus'),
    ('milk', 'lait'),
    ('bread', 'pain'),
    ('flower', 'fleur'),
    ('grape', 'raisin'),
    ('car', 'voiture'),
    ('truck', 'camion'),
]

country2capital = [ #The capital of Germany is Berlin.
    ('Germany', 'Berlin'),
    ('France', 'Paris'),
    ('China', 'Beijing'),
    ('the United States', 'Washington, D.C'),
    ('Italy', 'Rome'),
    ('Japan', 'Tokyo'),
    ('Russia', 'Moscow'),
    ('Spain', 'Madrid'),
    ('the United Kingdom', 'London'),
    ('Canada', 'Ottawa'),
    ('India', 'New Delhi'),
    ('Australia', 'Canberra'),
    ('Brazil', 'Brasília'),
    ('Mexico', 'Mexico City'),
    ('South Africa', 'Pretoria'),
    ('Egypt', 'Cairo'),
    ('Kenya', 'Nairobi'),
    ('South Korea', 'Seoul'),
    ('the Philippines', 'Manila'),
    ('Portugal', 'Lisbon'),
    ('Switzerland', 'Bern'),
    ('Thailand', 'Bangkok'),
    ('Turkey', 'Ankara'),
    ('Spain', 'Madrid'),
    ('Greece', 'Athens'),
]

# https://github.com/knowitall/chunkedextractor/blob/master/src/main/resources/edu/knowitall/chunkedextractor/demonyms.csv
demonyms = {country: resident for resident, country in csv.reader(open('demonyms.csv'))}
demonyms.update({'the United States': 'American', 'the United Kingdom': 'British', 'United States': 'American', 'United Kingdom': 'British'})
city2resident = [(capital, demonyms[country.replace('the ', '')]) for country, capital in country2capital]

grammar_correction = [
    ('Anna and Mike is going skiing.', 'Anna and Mike are going skiing.'),
    ('Anna and Pat are married; he has been together for 20 years.', 'Anna and Pat are married; they have been together for 20 years.'),
    ('I walk to the store and I bought milk.', 'I walked to the store and I bought milk.'),
    ('We all eat the fish and then made dessert.', 'We all ate the fish and then made dessert.'),
    ('Today I have went to the store to to buys some many bottle of water.', 'Today I went to the store to buy some bottles of water.'),
    ('I eated the purple berries.', 'I ate the purple berries.'),
    ('The patient was died.', 'The patient died.'),
    ('We think that Leslie likes ourselves.', 'We think that Leslie likes us.'),
    ('I have tried to hit ball with bat, but my swing is has miss.', 'I have tried to hit the ball with the bat, but my swing has missed.'),
]

drop_first_and_last = [
    ('4, 5, 0, 0', '5, 0'),
    ('3, 8, 3, 8, 3', '8, 3, 8'),
    ('4, 9, 4, 9, 4, 9, 9, 9, 9, 9', '9, 4, 9, 4, 9, 9, 9, 9'),
    ('5, 7, 7, 9, 8, 1, 4, 0, 6', '7, 7, 9, 8, 1, 4, 0'),
    ('2, 1, 1, 2, 2, 7, 2, 7', '1, 1, 2, 2, 7, 2'),
]

remove_two = [
    ('8, 0, 5, 12, 0, 2', '8, 5, 12, 2'),
    ('8, 19, 7, 8, 8, 8, 7, 7, 7, 7', '19'),
    ('0, 1, 18, 9, 9, 0, 15, 6, 1', '18, 15, 6'),
    ('0, 17, 4, 8, 4, 10, 1', '0, 17, 8, 10, 1'),
]

# https://stackoverflow.com/questions/20336524/verify-correct-use-of-a-and-an-in-english-texts-python
def starts_with_vowel_sound(word, pronunciations=cmudict.dict()):
    for syllables in pronunciations.get(word, []):
        return syllables[0][-1].isdigit() 

def add_a_or_an(word):
    word = lower(word)
    if word.endswith('s'): return f'a pair of {word}' # plies, scissors
    return ('an' if starts_with_vowel_sound(word) else 'a') + ' ' + word

class Relation(object):
    def __init__(self, _dict): self._dict = _dict
    def f(self, x): return self._dict.get(x, [])
    def inv_f(self, x): return self._inv_dict.get(x, [])
    # def el(self): return self._el
    def dom(self, xs=None): return set(self._dict.keys())
    def codom(self, ys=None): return set(join_lists(self._dict.values()))
    def b(self, x0, x1): return x1 in self._dict.get(x0, [])
    
class Set(object):
    def __init__(self, data, rel_names):
        # self.set_data(data)
        self.rel_names = rel_names
        for rel_name in self.rel_names:
            setattr(self, rel_name, Relation(_dict=defaultdict(list)))

class EqSet(Set):
    def __init__(self, data):
        self.data = data
        self.rel_names = ['equal']
        for rel_name, d in zip(self.rel_names, [{data[i]: [data[i]] for i in range(0, len(data))}]):
            setattr(self, rel_name, Relation(_dict=d))

class PoSet(Set):
    def __init__(self, data):
        self.data = data
        self.rel_names = ['prev', 'next', 'equal']
        for rel_name, d in zip(self.rel_names, [{data[i]: data[i - 1] for i in range(1, len(data))},
                                                {data[i]: data[i + 1] for i in range(0, len(data) - 1)},
                                                {data[i]: data[i] for i in range(0, len(data))}]):
            setattr(self, rel_name, Relation(_dict=d))

class SymSet(Set):
    def __init__(self, data):
        super().__init__(data, ['similar', 'opposite', 'equal'])
        for pair in data:
            for similars, opposites in [(pair[0], pair[1]), (pair[1], pair[0])]:
                for i, e in enumerate(similars):
                    self.equal._dict[e] = [e]
                    if len(similars) > 1: self.similar._dict[e] = list(set(similars) - {e})
                    if i == 0: self.opposite._dict[e] = opposites[:1]
        
class BijectSet(Set):
    def __init__(self, data):
        super().__init__(data, ['proj', 'inv_proj'])
        for a, b in data:
            self.proj._dict[a] = [b]
            self.inv_proj._dict[b] = [a]

class TreeSet(Set):
    def __init__(self, data):
        super().__init__(data, ['child', 'parent'])
        for parent, children in data.items():
            self.child._dict[parent] = children
            for child in children: self.parent._dict[child] = [parent]
        self.child._inv_dict, self.parent._inv_dict = self.parent._dict, self.child._dict

def MlM_gen(rels, cxt_len=3):
    candidates = OrderedDict()
    hop = 0; rel = rels[hop][0]
    query = choice(list(rel.dom()))
    candidates[hop] = [choice(r.f(query)) for r in rels[hop][:1]]
    # candidates[hop] = [choice(r.f(query)) for i, r in enumerate(rels[hop]) if i == 0 or random() > 0.5] # w/ distractors
    candidates[hop] += sample(list(rel.codom() - set(join_lists([r.f(query) for r in rels[hop]]))), 
        cxt_len - len(candidates[hop]))

    hop = 1; rel = rels[hop][0]
    # candidates[hop] = sample(list(rel.dom()), cxt_len)
    ans = choice(list(rel.codom()))
    candidates[hop] = [choice(rel.inv_f(ans))] + sample(list(rel.dom() - set(rel.inv_f(ans))), cxt_len - 1)
    cxt = sample(list(zip(*candidates.values())), cxt_len)

    def transform_fn(cxt, query):
        hop = 0; tgt, ans = seq(cxt).find(lambda x: rels[hop][0].b(query, x[0]))#[1]
        hop = 1; ans = rels[hop][0].f(ans)[0]
        return tgt, ans
    tgt, ans = transform_fn(cxt, query)
    hop = 1; candidates = ([rels[hop][0].f(x[1])[0] for x in cxt], [x[1] for x in cxt])
    return cxt, query, candidates, ans

def IlMlI_gen(rels, cxt_len=3):
    hop = 0
    query = choice(list(rels[hop][0].dom()))
    candidates0 = [choice(r.f(query)) for r in rels[hop][:1]]
    candidates0 += sample(list(rels[hop][0].codom() - set(join_lists([r.f(query) for r in rels[hop]]))), 
        cxt_len - len(candidates0))
    candidates0 = candidates0[:1] + sample(candidates0[1:], cxt_len - 1)

    hop = 1
    query1 = choice(list(rels[hop][0].dom()))
    candidates1 = [query1] + [choice(r.f(query1)) for r in rels[hop][:1]]
    candidates1 += sample(list(rels[hop][0].codom() - {query1} - set(join_lists([r.f(query1) for r in rels[hop]]))), 
        cxt_len - len(candidates1))
    # assert len(candidates1) == len(set(candidates1)), str(candidates1)
    cxt = sample(list(zip(candidates0, candidates1)), cxt_len)
    
    def transform_fn(cxt, query):
        hop = 0; ans = seq(cxt).find(lambda x: rels[hop][0].b(query, x[0]))[1]
        hop = 1; ans = seq(cxt).find(lambda x: x[0] != query and rels[hop][0].b(ans, x[1]))[0]
        hop = 2; ans = choice(rels[hop][0].f(ans))
        return ans
    ans = transform_fn(cxt, query)
    hop = 2; candidates = ([rels[hop][0].f(x[0])[0] for x in cxt], [x[0] for x in cxt])
    return cxt, query, candidates, ans

def g2c(g_fn, labels=['No', 'Yes', 'Maybe']):
    def wrapped(*args,**kwargs):
        cxt, query, candidates, ans = g_fn(*args,**kwargs)
        if tuple(candidates[0]) == tuple(candidates[1]):
            return (cxt, (query, choice(list(set(candidates[0]) - {query, ans}))), [labels], labels[1]) \
                if random() > 0.5 else (cxt, (query, ans), [labels], labels[0])
        else:
            ans_idx = candidates[0].index(ans)
            _ans = candidates[1][ans_idx]
            p = random()
            if p < 1/3: ans, label = ans, labels[0]
            elif 1/3 <= p < 2/3: ans, label = _ans, labels[1]
            else:
                if len(list(set(candidates[0] + candidates[1]) - {ans, _ans})) == 0:
                    print('empty', candidates[0], candidates[1], ans, _ans)
                ans, label = choice(list(set(candidates[0] + candidates[1]) - {ans, _ans})), labels[2]
            return cxt, (query, ans), [labels], label
    return wrapped

def inc(token):
    assert len(token) == 1 or token in ['->'], token
    if token.isalpha(): return chr(ord(token) + 1)
    elif token.isdigit(): return str(int(token) + 1)
    else: return token

def identity(x): return x
def upper(x): return x.upper()
def lower(x): return x.lower()
def x10(x): return x + '0'
def d10(x): return x[0]
def prepend(token): return lambda x: token + x

def to_cardinal(x): return digit2cardinal[x]
def to_ordinal(x): return digit2ordinal[x]
def to_digit(word):
    for d, w in digit2cardinal.items():
        if w == word: return d
    for d, w in digit2ordinal.items():
        if w == word: return d

def double(x): return x.upper() * 2
def single(x): return x[0]

def to_rand_letter(x): return choice(uppercases + lowercases)
def to_rand_digit(x): return choice(digits)


def inc(token):
    assert len(token) == 1 or token in ['->'], token
    if token.isalpha(): return chr(ord(token) + 1)
    elif token.isdigit(): return str(int(token) + 1)
    else: return token

def identity(x): return x
def upper(x): return x.upper()
def lower(x): return x.lower()
def x10(x): return x + '0'
def d10(x): return x[0]
def prepend(token): return lambda x: token + x

def to_cardinal(x): return digit2cardinal[x]
def to_ordinal(x): return digit2ordinal[x]
def to_digit(word):
    for d, w in digit2cardinal.items():
        if w == word: return d
    for d, w in digit2ordinal.items():
        if w == word: return d

def double(x): return x.upper() * 2
def single(x): return x[0]

def to_rand_letter(x): return choice(uppercases + lowercases)
def to_rand_digit(x): return choice(digits)

inverse_fns = {
    identity.__name__: identity, lower.__name__: upper, upper.__name__: lower, 
    double.__name__: single, x10.__name__: d10,
    to_cardinal.__name__: to_digit, to_ordinal.__name__: to_digit}
inverse_fns.keys()


from type import * 

class Task(object):
    def __init__(self, name, request, examples, features=None, cache=False):
        '''request: the type of this task
        examples: list of tuples of (input, output). input should be a tuple, with one entry for each argument
        cache: should program evaluations be cached?
        features: list of floats.'''
        self.cache = cache
        self.features = features
        self.request = request
        self.name = name
        self.examples = examples
        if len(self.examples) > 0:
            assert all(len(xs) == len(examples[0][0])
                       for xs, _ in examples), \
                "(for task %s) FATAL: Number of arguments varies." % name

    def __str__(self):
        if self.supervision is None:
            return self.name
        else:
            return self.name + " (%s)"%self.supervision

    def __repr__(self):
        return "Task(name={self.name}, request={self.request}, examples={self.examples}"\
            .format(self=self)

    def __eq__(self, o): return self.name == o.name

    def __ne__(self, o): return not (self == o)

    def __hash__(self): return hash(self.name)

    def describe(self):
        description = ["%s : %s" % (self.name, self.request)]
        for xs, y in self.examples:
            if len(xs) == 1:
                description.append("f(%s) = %s" % (xs[0], y))
            else:
                description.append("f%s = %s" % (xs, y))
        return "\n".join(description)

    def predict(self, f, x):
        for a in x:
            f = f(a)
        return f

    @property
    def supervision(self):
        if not hasattr(self, 'supervisedSolution'): return None
        return self.supervisedSolution

    def check(self, e, timeout=None):
        if timeout is not None:
            def timeoutCallBack(_1, _2): raise EvaluationTimeout()
        try:
            signal.signal(signal.SIGVTALRM, timeoutCallBack)
            signal.setitimer(signal.ITIMER_VIRTUAL, timeout)

            try:
                f = e.evaluate([])
            except IndexError:
                # free variable
                return False
            except Exception as e:
                eprint("Exception during evaluation:", e)
                return False

            for x, y in self.examples:
                if self.cache and (x, e) in EVALUATIONTABLE:
                    p = EVALUATIONTABLE[(x, e)]
                else:
                    try:
                        p = self.predict(f, x)
                    except BaseException:
                        p = None
                    if self.cache:
                        EVALUATIONTABLE[(x, e)] = p
                if p != y:
                    if timeout is not None:
                        signal.signal(signal.SIGVTALRM, lambda *_: None)
                        signal.setitimer(signal.ITIMER_VIRTUAL, 0)
                    return False

            return True
        # except e:
            # eprint(e)
            # assert(False)
        except EvaluationTimeout:
            eprint("Timed out while evaluating", e)
            return False
        finally:
            if timeout is not None:
                signal.signal(signal.SIGVTALRM, lambda *_: None)
                signal.setitimer(signal.ITIMER_VIRTUAL, 0)

    def logLikelihood(self, e, timeout=None):
        if self.check(e, timeout):
            return 0.0
        else:
            return NEGATIVEINFINITY

    @staticmethod
    def featureMeanAndStandardDeviation(tasks):
        dimension = len(tasks[0].features)
        averages = [sum(t.features[j] for t in tasks) / float(len(tasks))
                    for j in range(dimension)]
        variances = [sum((t.features[j] -
                          averages[j])**2 for t in tasks) /
                     float(len(tasks)) for j in range(dimension)]
        standardDeviations = [v**0.5 for v in variances]
        for j, s in enumerate(standardDeviations):
            if s == 0.:
                eprint(
                    "WARNING: Feature %d is always %f" %
                    (j + 1, averages[j]))
        return averages, standardDeviations

    def as_json_dict(self):
        return {
            "name": self.name,
            "request": str(self.request),
            "examples": [{"inputs": x, "output": y} for x, y in self.examples]
        }

def lcs(u, v):
    # t[(n,m)] = length of longest common string ending at first
    # n elements of u & first m elements of v
    t = {}

    for n in range(len(u) + 1):
        for m in range(len(v) + 1):
            if m == 0 or n == 0:
                t[(n, m)] = 0
                continue

            if u[n - 1] == v[m - 1]:
                t[(n, m)] = 1 + t[(n - 1, m - 1)]
            else:
                t[(n, m)] = 0
    l, n, m = max((l, n, m) for (n, m), l in t.items())
    return u[n - l:n]
    
def guessConstantStrings(task):
    if task.request.returns() == tlist(tcharacter):
        examples = task.examples
        guesses = {}
        N = 10
        T = 2
        for n in range(min(N, len(examples))):
            for m in range(n + 1, min(N, len(examples))):
                y1 = examples[n][1]
                y2 = examples[m][1]
                l = ''.join(lcs(y1, y2))
                if len(l) > 2:
                    guesses[l] = guesses.get(l, 0) + 1

        task.stringConstants = [g for g, f in guesses.items()
                                if f >= T]
    else:
        task.stringConstants = []
                    

    task.BIC = 1.
    task.maxParameters = 1

    task.specialTask = ("stringConstant",
                        {"maxParameters": task.maxParameters,
                         "stringConstants": task.stringConstants})

def loadPBETasks(directory="PBE_Strings_Track"):
    """
    Processes sygus benchmarks into task objects
    For these benchmarks, all of the constant strings are given to us.
    In a sense this is cheating
    Returns (tasksWithoutCheating, tasksWithCheating).
    NB: Results in paper are done without "cheating"
    """
    import os
    from sexpdata import loads, Symbol

    def findStrings(s):
        if isinstance(s, list):
            return [y
                    for x in s
                    for y in findStrings(x)]
        if isinstance(s, str):
            return [s]
        return []

    def explode(s):
        return [c for c in s]

    tasks = []
    cheatingTasks = []
    for f in os.listdir(directory):
        if not f.endswith('.sl'):
            continue
        with open(directory + "/" + f, "r") as handle:
            message = "(%s)" % (handle.read())

        expression = loads(message)

        constants = []
        name = f
        examples = []
        declarative = False
        for e in expression:
            if len(e) == 0:
                continue
            if e[0] == Symbol('constraint'):
                e = e[1]
                assert e[0] == Symbol('=')
                inputs = e[1]
                assert inputs[0] == Symbol('f')
                inputs = inputs[1:]
                output = e[2]
                examples.append((inputs, output))
            elif e[0] == Symbol('synth-fun'):
                if e[1] == Symbol('f'):
                    constants += findStrings(e)
                else:
                    declarative = True
                    break
        if declarative: continue
        
        examples = list({(tuple(xs), y) for xs, y in examples})

        task = Task(name, arrow(*[tstr] * (len(examples[0][0]) + 1)),
                    [(tuple(map(explode, xs)), explode(y))
                     for xs, y in examples])
        cheat = task

        tasks.append(task)
        cheatingTasks.append(cheat)

    for p in tasks:
        guessConstantStrings(p)
    return tasks, cheatingTasks

def retrieveJSONTasks(filename, features=False):
    """
    For JSON of the form:
        {"name": str,
         "type": {"input" : bool|int|list-of-bool|list-of-int,
                  "output": bool|int|list-of-bool|list-of-int},
         "examples": [{"i": data, "o": data}]}
    """
    with open(filename, "r") as f:
        loaded = json.load(f)
    TP = {
        "bool": tbool,
        "int": tint,
        "list-of-bool": tlist(tbool),
        "list-of-int": tlist(tint),
    }
    return [Task(
        item["name"],
        arrow(TP[item["type"]["input"]], TP[item["type"]["output"]]),
        [((ex["i"],), ex["o"]) for ex in item["examples"]],
        features=(None if not features else list_features(
            [((ex["i"],), ex["o"]) for ex in item["examples"]])),
        cache=False,
    ) for item in loaded]

'''
vocab = list(string.ascii_uppercase) #+ ['_'] * 16
# vocab = list(string.digits)[1:]
query_vocab = list(string.ascii_uppercase)
nrows, ncols = 12, 6
has_query, query_first = False, True
has_output = True
def map_fn(x): return x.lower()

old_stdout = sys.stdout
sys.stdout = mystdout = StringIO()
try:
    for _ in range(nrows):
#         input_tokens = sample(vocab, ncols)
        input_tokens = [choice(vocab)] * ncols
        i = random.randint(2, len(input_tokens) - 1)
        input_tokens[i] = choice(list(set(vocab) - {input_tokens[i]})) #'*' + input_tokens[i]
    #     input_tokens[-1] = input_tokens[0]
    #     special = choice(vocab)
    #     input_tokens = [choice(vocab).lower()] * (ncols - 1) + [special]
    #     shuffle(input_tokens)
        if not query_first: print(' '.join(input_tokens), end='')
        if has_query:
    #         query_tokens = sample(input_tokens, ncols - 1)
    #         query_tokens = input_tokens.copy()
            i = random.randint(0, len(input_tokens) - 2)
            query_tokens = [input_tokens[i]]
    #         query_tokens[i] = choice(vocab)
    #         query_tokens = [t.lower() for t in query_tokens]
#             print(',', ' '.join(query_tokens), end='')
            print('After', ' '.join(query_tokens), end=', ')
        if query_first: print(' '.join(input_tokens), end='')
        print(' -> ', end='')
        if has_output:
            ans_fn = identity #upper
            output_tokens = ans_fn(input_tokens[i])
    #         output_tokens = special.lower()
    #         output_tokens = choice(input_tokens)
    #         output_tokens = list(set(input_tokens) - set(query_tokens))
    #         output_tokens = map(map_fn, input_tokens)
    #         output_tokens = reverse(input_tokens)
    #         output_tokens = query_tokens[i].lower()
            print(''.join(output_tokens), end='')
        print()
finally: sys.stdout = old_stdout
example = mystdout.getvalue()
print(example)
texts['tmp'] = '\n' + example[:-1]
if has_query: ncols += 3



def vocab_fn(mix=False): return ''.join([v[0] for v in vocabs]) if mix else choice([v[0] for v in vocabs])

def input_fn(vocab, ncols):
    if vocab is None: vocab = vocab_fn()
    return sample(vocab, ncols)
#     return [choice(string.ascii_uppercase)] * ncols
def i_fn(ncols, i_range=[0, 0]):
    return choice(range(0 + i_range[0], ncols + i_range[1])) if type(i_range) in [list] else i_range

def get_tgt(token, tgt_vocab, tgt_fn=identity):
    if tgt_vocab is None: return tgt_fn(token)
    return choice(tgt_vocab)
#     return choice(list(set(tgt_vocab) - {token}))

def update_input(input_tokens, i, tgt, updates=[]):
    for offset, val in updates: input_tokens[i + offset] = val

def gen_query(input_tokens, i):
    return None, None
    return [token for j, token in enumerate(input_tokens) if j != i], None
    return [input_tokens[i - 1]], None

def get_ans_fn(vocab):
    return identity
    fns = []
    vocab = set(vocab)
    if vocab.issubset(set(string.ascii_uppercase)): fns += [double]#, lower]
    if vocab.issubset(set(string.ascii_lowercase)): fns += [double]#, upper]
    if vocab.issubset(set(string.digits)):
#         fns += [prepend('0'), prepend('1'), double, x10]
        fns += [to_word, to_ordinal]
    if len(fns) == 0: fns =[identity]
    return choice(fns)

task_configs = {
    'ith': {'i_range': 1},
    'after bracket': {'i_range': [1, 0], 'updates': [(-1, '(')]},
    'in brackets': {'i_range': [1, -1], 'updates': [(-1, '('), (1, ')')]},
    'special type': {'vocab': vocabs[0][0], 'tgt_vocab': vocabs[1][0], 'ans_fn': upper},
    'special': {'i_range': [2, 0], 'use_extended_vocab': True, 'vocab_per_row': True}
}

def gen_example(nrows=16, ncols=3, use_extended_vocab=False, vocab_per_row=False,
                vocab=None, tgt_vocab=None, tgt_fn=identity, ans_fn=identity, 
                i_range=[0, 0], updates=[]):
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    try:
#         vocab = vocab_fn()
        ans_fn = ans_fn or get_ans_fn(tgt_vocab)
        if use_extended_vocab and not vocab_per_row:
            vocab, tgt_fn, ans_fn = get_extended_vocab_and_fns(base_vocab=None, fn=None, tgt_fn1=None)#, ans_fn=None)
        for _ in range(nrows):
            if use_extended_vocab and vocab_per_row:
                vocab, tgt_fn, ans_fn = get_extended_vocab_and_fns(base_vocab=None, fn=None, tgt_fn1=None)#, ans_fn=None)
#             random.seed(datetime.now())
            input_tokens = input_fn(vocab, ncols)
            i = i_fn(ncols, i_range=i_range)
            tgt = get_tgt(input_tokens[i], tgt_vocab, tgt_fn=tgt_fn)
            input_tokens[i] = tgt
            update_input(input_tokens, i, tgt, updates=updates)
            ans = ans_fn(tgt)
            query = gen_query(input_tokens, i)
            if query[0]: print(' '.join(query[0]), end=', ')
    #         print()
            print(' '.join(input_tokens), end='')
            if query[1]: print(',\n' + ' '.join(query[1]), end='')
            print(' ->', end=' ')
            print(ans)
    finally: sys.stdout = old_stdout
    return mystdout.getvalue(), ans_fn

task_name = 'after bracket'
# task_name = 'in brackets'
task_name = 'ith'
# task_name = 'special type'
configs = task_configs[task_name]
# configs = list(task_configs.items())[-1][1]
nrows, ncols = 12, 4
vocab = None or upper_letters
tgt_fn, ans_fn = identity, identity
# tgt_fn, ans_fn = lower, upper
example, ans_fn = gen_example(nrows=nrows, ncols=ncols, 
                              vocab=upper_letters, 
                              tgt_fn=tgt_fn, ans_fn=ans_fn,
                              **configs)
print(example)
texts['tmp'] = '\n' + example[:-1]
'''
