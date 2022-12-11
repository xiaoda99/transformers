import sys
import os
import json
import csv
from collections import defaultdict, OrderedDict, Counter, Iterable
from functools import partial, wraps
import string
from random import choice, choices, shuffle, sample, randint, random, seed
from dataclasses import dataclass, fields
import traceback
import time
import re
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn.functional as F 

from common_utils import join_lists, my_isinstance, lget, fn2str
from openai_utils import query_openai

sys.path.insert(0, '/nas/xd/projects/PyFunctional')
from functional import seq
from functional.pipeline import Sequence

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

# posets = [list(string.ascii_uppercase)[:14], list(string.ascii_lowercase)[:14], list(string.ascii_uppercase)[14:], list(string.ascii_lowercase)[14:], digits, cardinals, ordinals,
#     times_of_day, days_of_week, months, seasons, ages_of_life, times_of_history, #units_of_time, 
#     things, sizes]# units_of_length, units_of_mass, SI_prefixes_small, SI_prefixes_large]
posets = [times_of_day,clock_of_day,years, days_of_week, months, seasons]
closed_posets = [list(string.ascii_uppercase)[:7], list(string.ascii_lowercase)[:7],][:] + [digits, cardinals, #ordinals[:5], 
    days_of_week, months, ]#seasons, times_of_history, ages_of_life, sizes]
open_posets = [times_of_day, ages_of_life, times_of_history, units_of_length, units_of_mass, things, sizes, ]

def does2did():
    d = {vt.does: [vt.did] for vt in verb_tenses()}
    d['sings'] = ['sang']; d['leaves'] = ['left']
    return d

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
    ('healthy', 'ill'),
    ('male', 'female'),
    ('man', 'woman'),
    # ('true', 'false'),
]

'''gpt-3 prompt
Types of tools: hammer, screwdriver, saw, drill, wrench
Types of clothes: shirt, pants, dress, coat, shoes
Types of fruits: apple, grape, pear, banana, orange
Types of animals: dog, cat, horse, rabbit, pig'''
def types_of_things(): return {
    'animal': ['chicken', 'duck', 'goose', 'dog', 'lion', 'cow', 'donkey', 'horse', 'sheep', 'goat', 'bear', 'tiger', 'cat', 'zebra', 'pig', 
            'giraffe', 'monkey', 'rabbit', 'elephant', 'wolf', 'lion', 'deer', 'fox', 'gorilla', 'kangaroo', 'squirrel', 'mouse'], # 21-5, 15-8
    'fruit': ['apple', 'banana', 'pear', 'grapes', 'cherry', 'orange', 'peach', 'plum', 'lemon', 'mango', 'blackberry',
            'blueberries', 'strawberries', 'durian', 'papaya', 'watermelon', 'pineapple', 'kiwi', 'apricot', 'lime'], # may be food too?
    'drink': ['tea', 'coffee', 'beer', 'wine', 'whiskey', 'vodka', 'soda', 'juice', 'cocktail'],  # some as alcohol, 21-5, 15-8
    'food': ['hamburger', 'burger', 'bread', 'meat', 'pizza', 'cake', 'steak', 'spaghetti',
            # 'biscuits', 'spaghetti', 'chips', 'peanuts', 'nuts', 'pork', 'beef', 'mutton'
            ],  # last three as meat， 21-5， 15-8
    'weapon': ['gun', 'handgun', 'shotgun', 'rifle',  'pistol', 'revolver', 'grenade', 'cannon'], #'bomb', 'dagger', 'sword',], # 21-5, 15-8, though latter prefers firearm
    'color': ['white', 'black', 'red', 'yellow', 'blue', 'green', 'purple', 'pink', 'gray'],  # 15-8
    # 'insect': ['bee', 'ant', 'mosquito', 'wasp', 'butterfly', 'beetle', 'spider'],  # , 'fly'
    # 'flower': ['rose', 'tulip', 'lily', 'daisy', 'sunflower'],
    # 'vehicle': ['car', 'bus', 'tractor', 'airplane', 'ship', 'bicycle', 'truck', 'train', 'motorbike', 'helicopter', 'carriage', 
    #             'subway', 'taxi', 'van', 'boat'],  # transportation
    # 'furniture': ['sofa', 'couch', 'desk', 'chair', 'table', 'bed', 'bookshelf'],# 'closet', 'wardrobe'],
    # 'tool': ['hammer', 'spanner', 'awl', 'scissors', 'saw', 'shovel', 'screwdriver', 'wrench', 'drill', 'pliers'], #, 'axe' should be weapon?
    # 'clothing': ['shirt', 'pants', 'dress', 'coat', 'socks', 'hat', 'tie', 'jacket', 'skirt', 'trousers', 'jeans'], #, 'shoes'
    # 'appliance': ['microwave', 'fridge', 'washer', 'dryer', 'washing machine'],  #, 'oven'
    # 'fish': [],
    # 'country': [],
    # 'language': [],
    # 'temperature': [],
    # 'age': [],
    # 'sport': [],
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

_capabilities_of_things = [ # A x can y.
    # avoid the hammer->hammer, saw->saw, glider->glide pattern
    ('knife', 'cut'),
    ('calculator', 'add'), # calculate
    ('phone', 'call'),
    ('printer', 'print'),
    ('pen', 'write'),
    ('pencil', 'write'),
    ('saw', 'cut'),  # saw
    ('oven', 'bake'),
    ('pot', 'cook'), # boil
    ('gun', 'shoot'),
    ('dagger', 'stab'),
    # ('pan', 'fry'),
    ('shovel', 'dig'),
    # ('hammer', 'hammer'),
    ('axe', 'chop'),
    # ('drill', 'drill'),  # bore
    ('lamp', 'light'),
    ('fan', 'cool'),  # blow
    ('washing machine', 'wash'),  # or washer
    ('opener', 'open'),  # or washer
    # ('dryer', 'dry'),
    # ('lighter', 'light'),
    
    ('TV', 'watch'),  # show
    ('car', 'drive'),

    ('plane', 'fly'),
    ('bicycle', 'ride'),
    ('glider', 'fly'),  # glide
    # ('skateboard', 'skate'),
    ('piano', 'play'),
    ('violin', 'play'),
    # ('book', 'read'), # teach
]

def capabilities_of_things():
    d = defaultdict(list)
    for thing, cap in _capabilities_of_things: d[cap].append(thing)
    return d

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

_en2fr = [
    ('apple', 'pomme'),
    ('banana', 'banane'),
    ('pear', 'poire'),
    # ('grapes', 'raisins'),
    # ('watermelon', 'pastèque'),
    ('butter', 'beurre'),
    ('cheese', 'fromage'),
    ('cat', 'chat'),
    ('dog', 'chien'),
    ('pig', 'cochon'),
    # ('bear', 'ours'),
    ('sugar', 'sucre'),
    ('coffee', 'café'),
    ('tea', 'thé'),
    ('juice', 'jus'),
    ('milk', 'lait'),
    ('bread', 'pain'),
    ('flower', 'fleur'),
    ('car', 'voiture'),
    ('truck', 'camion'),
    ('book', 'livre'),
    ('knife', 'couteau'),
]

def en2fr(): return {en: [wrap_noun_to_french(en)] for en, _ in _en2fr}

_country2capital = [ #The capital of Germany is Berlin.
    ('Germany', 'Berlin'),
    ('France', 'Paris'),
    ('China', 'Beijing'),
    ('the United States', 'Washington, D.C'),
    ('Italy', 'Rome'),
    ('Japan', 'Tokyo'),
    ('Russia', 'Moscow'),
    ('Spain', 'Madrid'),
    ('England', 'London'),
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
    ('Greece', 'Athens'),
]

def country2capital():  # convert to same form as TreeSet types_of_things
    return {country: [capital] for country, capital in _country2capital}


def city2resident():
    if not hasattr(city2resident, 'demonyms'):
        # https://github.com/knowitall/chunkedextractor/blob/master/src/main/resources/edu/knowitall/chunkedextractor/demonyms.csv
        city2resident.demonyms = {country: resident for resident, country in csv.reader(open('demonyms.csv'))}
        city2resident.demonyms.update({'the United States': 'American', 'the United Kingdom': 'British', 'England': 'English'})
    return {capital: city2resident.demonyms[country.replace('the ', '')] for country, capital in _country2capital}

def wrap_noun(noun):
    prompt_fn = lambda s: \
f'''apple: There is an apple.
chip: There are chips.
coffee: There is coffee.
biscuit: There are biscuits.
dog: There is a dog.
tea: There is tea.
{s}: There'''
    def extract_fn(text):
        text = text.replace(' is ', '').replace(' are ', '')
        if text.endswith('.'): text = text[:-1]
        if not text.split()[-1].startswith(noun[:2]): # e.g. 'red' -> 'a red apple'
            # print(f'{noun} -> {text}. Skip abnormal wrap')
            text = noun
        return text
    return extract_fn(query_openai(prompt_fn(noun)))

def wrap_noun_to_french(noun):
    prompt_fn = lambda s: \
f'''dog: Martin a un chien.
apple: Martin a une pomme.
tea: Martin a du thé.
{s}: Martin a'''
    def extract_fn(text):
        assert text.endswith('.'); text = text[:-1]
        return text.strip()  # strip leading spaces
    return extract_fn(query_openai(prompt_fn(noun))) if noun != 'tea' else 'du thé'

def wrap_noun2(noun):
    prompt_fn = lambda s: \
f'''summer: Martin arrived in the summer.
Friday: Martin arrived on Friday.
morning: Martin arrived in the morning.
February: Martin arrived in February.
noon: Martin arrived at noon.
2020: Martin arrived in 2020.
{s}: Martin arrived'''
    # print(query_openai(prompt_fn(noun)))
    def extract_fn(text):
        text = text.strip()
        if text.endswith('.'): text = text[:-1]
        if not text.split()[-1].startswith(noun[:2]): # e.g. 'red' -> 'a red apple'
            # print(f'{noun} -> {text}. Skip abnormal wrap')
            text = noun
        return text
    # print(query_openai(prompt_fn(noun)))
    return extract_fn(query_openai(prompt_fn(noun)))

def conj(positive=True):
    ss = ["", "So", "Therefore,", "Yes,"]
    s = choice(ss)
    return s if s == "" else s + " "

def xxx_be(positive=True):
    ss = [
        ["is"],
        ["definitely is", "sometimes is", "always is"],
        ["may be", "must be", "should be"],
        ["may sometimes be", "must always be", "should always be", "has always been"],
        [", in all respects, is", ", anyhow, is", ", as far as I know, is", ", generally speaking, is"],
    ]
    s = choice(choice(ss))
    return s if s.startswith(",") else " " + s
    
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

class Relation(object):
    def __init__(self, name, _dict):
        self.name = name
        self._dict = _dict
        self._inv_dict = None
        self.inv_rel = None

    def f(self, x): return self._dict.get(x, [])
    def inv_f(self, x): return self._inv_dict.get(x, [])
    # def el(self): return self._el
    def dom(self, xs=None): return set(self._dict.keys())
    def codom(self, ys=None): return set(join_lists(self._dict.values()))
    def b(self, x0, x1): return x1 in self._dict.get(x0, [])
    
class Set(object):
    def __init__(self, data, rel_names):
        self.data = data
        self.rel_names = rel_names
        for rel_name in self.rel_names:
            setattr(self, rel_name, Relation(name=rel_name, _dict=defaultdict(list)))

    def use(self, rel_names):
        if isinstance(rel_names, str): rel_names = [rel_names]
        self.used_rel_names = rel_names
        self.relations = [getattr(self, rel_name) for rel_name in rel_names]
        return self

    def __str__(self):
        return f"{self.data.__name__}.{self.__class__.__name__}.{'|'.join(self.used_rel_names)}"

class EqSet(Set):
    def __init__(self, data):
        super().__init__(data, ['equal'])
        data = data()
        for rel_name, d in zip(self.rel_names, [{data[i]: [data[i]] for i in range(0, len(data))}]):
            setattr(self, rel_name, Relation(name=rel_name, _dict=d))
        self.equal._inv_dict = self.equal._dict
        self.equal.inv_rel = self.equal

class PoSet(Set):
    def __init__(self, data):
        super().__init__(data, ['prev', 'next', 'equal'])
        data = data()
        vector = choice(data)
        # for vector in data:
        for rel_name, d in zip(self.rel_names, [{vector[i]: [vector[i - 1]] for i in range(1, len(vector))},
                                                {vector[i]: [vector[i + 1]] for i in range(0, len(vector) - 1)},
                                                {vector[i]: [vector[i]] for i in range(0, len(vector))}]):
            setattr(self, rel_name, Relation(name=rel_name, _dict=d))
        self.prev._inv_dict, self.next._inv_dict = self.next._dict, self.prev._dict
        self.equal._inv_dict = self.equal._dict

class SymSet(Set):
    def __init__(self, data):
        super().__init__(data, ['similar', 'opposite', 'equal'])
        data = data()
        for pair in data:
            for similars, opposites in [(pair[0], pair[1]), (pair[1], pair[0])]:
                for i, e in enumerate(similars):
                    self.equal._dict[e] = [e]
                    if len(similars) > 1: self.similar._dict[e] = list(set(similars) - {e})
                    if i == 0: self.opposite._dict[e] = opposites[:1]
        self.opposite._inv_dict, self.similar._inv_dict = self.opposite._dict, self.similar._dict
        self.equal._inv_dict = self.equal._dict
        
class BijectSet(Set):  # can be treated as a special case of TreeSet?
    def __init__(self, data):
        super().__init__(data, ['proj', 'inv_proj', 'equal'])
        data = data()
        for a, b in data:
            self.proj._dict[a] = [b]
            self.inv_proj._dict[b] = [a]
            self.equal._dict[a] = [a]
            self.equal._dict[b] = [b]
        self.proj._inv_dict, self.inv_proj._inv_dict = self.inv_proj._dict, self.proj._dict
        self.equal._inv_dict = self.equal._dict

class TreeSet(Set):
    def __init__(self, data):
        super().__init__(data, ['child', 'parent', 'equal'])
        data = data()
        for parent, children in data.items():
            self.child._dict[parent] = children
            # self.equal._dict[parent] = [parent]
            for child in children:
                self.parent._dict[child] = [parent]
                self.equal._dict[child] = [child]
        self.child._inv_dict, self.parent._inv_dict = self.parent._dict, self.child._dict
        self.equal._inv_dict = self.equal._dict
        self.child.inv_rel, self.parent.inv_rel = self.parent, self.child
        self.equal.inv_rel = self.equal

def enumerate_sample(cxt_len, rel):
    return list(range(cxt_len)), sample(list(rel.dom()), cxt_len)

def grouped_sample(cxt_len, rel, n_groups=2, reverse=False, min_group_size=None, max_group_size=None):
    group_sizes = []
    while not group_sizes or (min_group_size and min(group_sizes) != min_group_size) \
                        or (max_group_size and max(group_sizes) != max_group_size):
        cut_points = sorted(sample(range(1, cxt_len - 1), n_groups - 1))
        group_sizes = [stop - start for start, stop in zip([0] + cut_points, cut_points + [cxt_len])]
        group_sizes = sorted(group_sizes, reverse=reverse)
        # print('In grouped_sample:', group_sizes)  # debug
    groups = sample(list(rel.dom()), n_groups)
    cxt = join_lists([list(zip([size] * size, sample(rel.f(group), size) if rel.name != 'equal' else [group] * size))
                    for group, size in zip(groups, group_sizes)])
    return tuple(map(list, zip(*cxt)))

def swap(l, dst, src=0):
    if dst != src: l[dst], l[src] = l[src], l[dst]
    return l

def distractive_sample(cxt_len, rel, ans_i=0):
    query = choice(list(rel.dom()))
    ans = choice(rel.f(query))
    distractors = sample(list(rel.codom() - set(rel.f(query))), cxt_len - 1)
    distractors0 = [rel.inv_f(x)[0] for x in distractors]
    return tuple([swap(l, ans_i) for l in [[query] + distractors0, [ans] + distractors]])
 
def negate(bool_fn):
    @wraps(bool_fn)  # https://stackoverflow.com/questions/42561843/python-negate-boolean-function
    def wrapperd_fn(*args, **kwargs): return not bool_fn(*args, **kwargs)
    return wrapperd_fn

def MlM_gen(rels, cxt_len=3, cxt_sample_fn=None, query=None, has_local_hop=True, do_negate=False):
    rels = [s.relations for s in rels]
    candidates = OrderedDict()
    fixed_query = query is not None
    position_relevant = cxt_sample_fn is not None and cxt_sample_fn.__name__ == 'enumerate_sample'
    
    hop = 0; rel = rel0 = rels[hop][0]
    bool_fn0, (candidates[hop - 1], candidates[hop]) = (rel.b, distractive_sample(cxt_len, rel)) \
        if not fixed_query else (lambda x, y: x == y, cxt_sample_fn(cxt_len, rel))
    if not fixed_query: query = candidates[hop - 1][0]
    elif not position_relevant: assert query == candidates[hop - 1][0], f'{query} != {candidates[hop - 1][0]}'

    hop = 1; rel = rel1 = rels[hop][0]
    candidates[hop], candidates[hop + 1] = distractive_sample(cxt_len, rel.inv_rel)[::-1] \
        if has_local_hop else (candidates[hop - 1].copy(), [rel.f(x)[0] for x in candidates[hop - 1]])
    
    if do_negate:
        assert cxt_len == 2; bool_fn0 = negate(bool_fn0)
        for h in [hop, hop + 1]: candidates[h] = candidates[h][::-1]

    tuples = list(zip(*candidates.values()))
    if not position_relevant: shuffle(tuples)
    cxt = [t[int(not fixed_query):3] for t in tuples]
    candidates = tuple(list(c) for c in zip(*tuples))

    def transform_fn(cxt, query):
        *_, tgt, ans = seq(cxt).find(lambda x: bool_fn0(query, x[0])); chain = (tgt, ans)
        ans = rel1.f(ans)[0]; chain += (ans,)
        return chain
    ans_chain = transform_fn(cxt, query)
    if fixed_query: cxt, query = [x[1:] for x in cxt], None
    if not has_local_hop: cxt = [x[0] for x in cxt]
    return cxt, query, candidates, ans_chain

def MlMlM_gen(rels, cxt_len=3):
    rels = [s.relations for s in rels]
    candidates = OrderedDict()
    
    hop = 0; rel = rel0 = rels[hop][0]; bool_fn0 = rel.b
    candidates[hop - 1], candidates[hop] = distractive_sample(cxt_len, rel)
    query = candidates[hop - 1][0]

    hop = 1; rel = rel1 = rels[hop][0]; bool_fn1 = rel.b
    (query1, *_), candidates[hop] = distractive_sample(cxt_len - 1, rel)
    candidates[hop] = [query1] + candidates[hop]

    hop = 2; rel = rel2 = rels[hop - 2][0]
    candidates[hop] = candidates[hop - 2]
    candidates[hop + 1] = [rel.f(x)[0] for x in candidates[hop]]

    tuples = list(zip(*candidates.values())); shuffle(tuples)
    cxt = [t[1:3] for t in tuples]
    candidates = tuple(list(c) for c in zip(*tuples))

    def transform_fn(cxt, query):
        tgt, ans = seq(cxt).find(lambda x: bool_fn0(query, x[0])); chain = (tgt, ans)
        ans, _ = seq(cxt).find(lambda x: x[0] != query and bool_fn1(ans, x[1])); chain += (ans,)
        ans = rel2.f(ans)[0]; chain += (ans,)
        return chain
    return cxt, query, candidates, transform_fn(cxt, query)

def swap_qa(g_fn):
    def wrapped(*args,**kwargs):
        cxt, query, candidates, ans_chain = g_fn(*args,**kwargs)
        (query, *ans_chain) = (query, *ans_chain)[::-1]
        return cxt, query, candidates[::-1], tuple(ans_chain)
    return wrapped

def g2c(g_fn, cls_labels=['Yes', 'No']):
    def wrapped(*args,**kwargs):
        cxt, query, candidates, (*a, ans0, ans) = g_fn(*args,**kwargs)
        (_ans0, _ans), label = ((ans0, ans), cls_labels[0]) if random() < 0.5 else \
            (choice([(c0, c) for q, *_, c0, c in zip(*candidates) if c != ans and q != query]), cls_labels[1])
        return cxt, query, candidates, (*a, _ans0, _ans), label
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
    # wrapped.__name__ = f'g2c({g_fn.__name__})'
    return wrapped

def _str(l, vocab=None, sep=' '):
    if l is None: return ''
    if isinstance(l, str) or not isinstance(l, Iterable): l = [l]
    l = [e for e in l if not my_isinstance(e, Sequence)] #type(e).__name__ != 'Sequence']
    if isinstance(l, (dict, OrderedDict)): l = [f'{k}: {v}' for k, v in l.items()]
    return sep.join(str(i) for i in l)

def options2str(options): return '[' + ' | '.join(options) + ']'  # ' or '.join(options) + '?'

def make_examples(task, nrows=4, vocab_for_each_row=True, **kwargs):
    vocab_fn, example_gen_fn = task[:2]
    vocabs, examples = [], []
    qa_set = set() # for dedup
    if not vocab_for_each_row: vocab = vocab_fn()
    for i in range(nrows * 2):
        if vocab_for_each_row: vocab = vocab_fn()
        cxt, query, candidates, ans, *a = example_gen_fn(vocab, **kwargs)
        if isinstance(query, list): query = tuple(query)
        if (tuple(cxt), query, ans) not in qa_set:
            qa_set.add((tuple(cxt), query, ans))
            vocabs.append(vocab)
            examples.append([cxt, query, candidates, ans, *a])
        if len(examples) == nrows: break
    return vocabs, examples

def _item2str(item, vocab=None, reverse=False):
    return (f'{item[1]} {item[0]}' if reverse else f'{item[0]} {item[1]}') if isinstance(item, tuple) else f'{item}'

def _cxt2str(cxt, vocab=None, prefix='', suffix='', sep=', ', item2str=_item2str):
    return prefix + sep.join([item2str(item, vocab) for item in cxt]) + suffix

@dataclass
class Ranges:
    bos: tuple = None
    ans: tuple = None
    ans0: tuple = None
    query: tuple = None
    tgt: tuple = None
    candidates: tuple = None
    example: tuple = None

# adapted from find_token_range in https://github.com/kmeng01/rome/blob/main/experiments/causal_trace.py
def locate(tokens, substring, return_last=False):
    if substring is None: return None
    whole_string = "".join(t for t in tokens)
    assert substring in whole_string, f'{tokens}\n{substring} not in {whole_string}'
    if substring.strip() in ['->', '?']:
        index_fn = getattr(whole_string, 'index' if not return_last else 'rindex')
        char_loc = index_fn(substring)
    else:
        try: matches = list(re.finditer(r"\b%s\b" % substring, whole_string))
        except Exception: print(f'sub = {substring}, whole = {whole_string}'); raise
        assert len(matches) > 0, f'{tokens}\n{substring} not match {whole_string}'
        char_loc = matches[-int(return_last)].span()[0]
    loc = 0; tok_start, tok_end = None, None
    for i, t in enumerate(tokens):
        loc += len(t)
        _t = t[1:] if t.startswith(' ') else t
        if tok_start is None and loc > char_loc:
            assert substring.find(_t) in [0, 1], \
                f'{whole_string}\n{tokens}\n{substring} not startswith {_t} at {i}. loc = {loc}, char_loc = {char_loc}'
            tok_start = i
        if tok_end is None and loc >= char_loc + len(substring):
            assert substring.endswith(_t), f'{whole_string}\n{tokens}\n{substring} not endswith {_t} at {i}'
            tok_end = i + 1
            break
    return (tok_start, tok_end)

def example2ranges(example, tokens, bos_token):
    cxt, query, candidates, (tgt, *_, ans0, ans), *cls = example
    return Ranges(
        bos = locate(tokens, bos_token, return_last=True),
        ans = locate(tokens, ans, return_last=True),
        ans0 = locate(tokens, ans0),
        query = locate(tokens, query, return_last=True),
        tgt = locate(tokens, tgt),
        candidates = tuple(map(np.array, zip(*[locate(tokens, cand) for cand in candidates[1]]))) if candidates else None,
        example = (0, len(tokens))
    )

def move_ranges(r, offset):
    for field in fields(r):
        name = field.name; pair = getattr(r, name)
        if pair is not None: setattr(r, name, tuple([i + offset for i in pair]))
    return r

def locate_ranges(examples, example_strs, tokenizer, bos_token):
    ranges, all_tokens, newline_token = [], [], tokenizer.tokenize('\n')[0]  # 'Ċ'
    assert len(examples) == len(example_strs)
    for i, (e, e_str) in enumerate(zip(examples, example_strs)):
        # tokens = tokenizer.tokenize(e_str)  # can not work with locate
        tokens = [tokenizer.decode([i]) for i in tokenizer.encode(e_str)]
        assert ''.join(tokens) == e_str, f"{tokens} -> {''.join(tokens)} != {e_str}"
        r = example2ranges(e, tokens, bos_token[i] if isinstance(bos_token, (tuple, list)) else bos_token)
        ranges.append(move_ranges(r, len(all_tokens)))
        all_tokens += tokens + [newline_token]
    return ranges

abstract_bos_token = ' ->'

def locate_answers(input_ids, tokenizer, bos_indices=None, bos_token=None, eos_token='Ċ',
        space_token='Ġ', nrows=None):
    assert input_ids.size(0) == 1  # bsz == 1
    if bos_indices is None:
        bos_id = tokenizer.convert_tokens_to_ids(bos_token.replace(' ', space_token))
        bos_indices = (input_ids[0] == bos_id).nonzero().squeeze(1).tolist()#[1:]
    if nrows is not None:
        assert nrows == len(bos_indices)
    else:
        nrows = len(bos_indices)
    if eos_token is not None:
        eos_id = tokenizer.convert_tokens_to_ids(eos_token)
        eos_indices = (input_ids[0] == eos_id).nonzero()[-nrows:].squeeze(1).tolist()
    else:
        # eos_indices = bos_indices[1:] + [input_ids.size(1)]
        eos_indices = [bos_i + 2 for bos_i in bos_indices]
    # labels = torch.ones(input_ids.size(0), input_ids.size(1) - 1).long() * (-100)
    labels = torch.ones_like(input_ids) * (-100)
    answers = []
    for bos_i, eos_i in zip(bos_indices, eos_indices):
        ans_ids = input_ids[0, bos_i + 1: eos_i]
        labels[0, bos_i: eos_i - 1] = ans_ids
        answers.append(ans_ids.numpy())
    return bos_indices, eos_indices, answers, labels

# bos_token='▁is'; eos_token='</s>' for s2s
# bos_token='Ġ->', eos_token='Ċ' for gpt
def make_data_tuple(text, examples, tokenizer, k_shot=3, bos_token=' ->', eos_token=None, s2s=False):
    example_strs = text.strip().split('\n')
    input_ids = tokenizer.encode(text, return_tensors='pt')
    ranges = locate_ranges(examples, example_strs, tokenizer, bos_token)
    bos_indices = [r.bos[0] for r in ranges]
    bos_indices, eos_indices, answers, labels = locate_answers(input_ids, tokenizer, bos_indices=bos_indices, eos_token=eos_token)
    if s2s:  # for t5 models
        bos_i, eos_i = bos_indices[-1], eos_indices[-1]
        assert eos_i == input_ids.size(1) - 1, f'{eos_i} != {input_ids.size()}[1] - 1'
        assert tokenizer.convert_ids_to_tokens(input_ids[0, -1].item()) == eos_token == '</s>', \
            f"{tokenizer.convert_ids_to_tokens(input_ids[0, -1].item())} != '</s>'"
        input_ids = torch.cat([input_ids[:, : bos_i + 1], input_ids[:, -1:]], dim=1) # append trailing '</s>'
        answers, labels = answers[-1:], labels[:, bos_i: eos_i - 1]
        bos_indices, eos_indices = [bos_i - bos_i], [eos_i - bos_i]
    else:
        labels[:, :bos_indices[k_shot]] = -100  # 只算k_shot个示例后的loss
    return input_ids, labels, ranges, example_strs, bos_indices, eos_indices, answers

def query2wh(vocab, query2str):
    wh = 'who'
    if query2str(wh, vocab).startswith(wh): wh = wh.capitalize()
    return wh

def make_input_str(task, vocabs, examples, abstract=0, do_swap_qa=False, options_position=None):
    cxt_len = len(examples[0][0])
    if abstract != 0:
        item2str, query2str = (partial(_item2str, reverse=abstract == 2), _str)
        if cxt_len == 1:
            item2str, query2str = (lambda i, _: f'{i[1]}', lambda q, _: '')
            examples = [(cxt, None, None, (None, ans0, ans), *cls) for cxt, query, options, (tgt, ans0, ans), *cls in examples]
        cxt2str, bos_token, ans2str = partial(_cxt2str, item2str=item2str), abstract_bos_token, _str
    else:
        cxt2str, query2str, bos_token, ans2str = [lget(task, i, '?' if i == 4 else _str) for i in range(2, 6)]
    def example2str(vocab, example):
        cxt, query, candidates, (*_, ans), *cls = example
        if do_swap_qa: query, ans, real_ans = query2wh(vocabs[0], query2str), query, ans
        strs = [cxt2str(cxt, vocab), query2str(query, vocab)]
        if options_position is not None: strs.insert(options_position, options2str([c[-1] for c in candidates]))
        s = '. '.join(s for s in strs if s != '') + bos_token + ' ' + ans2str(ans)
        if bos_token == '':
            query_str = strs[1]; assert query_str != ''
            _bos_token = "'s" if query_str.endswith("'s") else query_str.split()[-1]
        else:
            _bos_token = bos_token
        if do_swap_qa: _bos_token = '?'; s += _bos_token + ' ' + ans2str(real_ans)
        if len(cls) > 0: _bos_token = '?'; s += _bos_token + ' ' + _str(cls[0]) # g2c
        return s, _bos_token
    example_strs, bos_tokens = zip(*[example2str(v, e) for v, e in zip(vocabs, examples)])
    return examples, '\n'.join(example_strs) + '\n', bos_tokens

def get_answer_index(example):
    cxt, query, cands, (*_, ans), *cls = example
    return cands[-1].index(ans)

def generate(task, do_negate=False, do_swap_qa=False, do_g2c=False, nrows=8, cxt_len=3, abstract=0, plot=True, verbose=True):
    # unpack, modify and repack to avoid in-place modification of task
    if do_negate: vocab_fn, gen_fn, *a = task; task = (vocab_fn, partial(gen_fn, do_negate=True), *a)
    if do_swap_qa: vocab_fn, gen_fn, *a = task; task = (vocab_fn, swap_qa(gen_fn), *a)
    if do_g2c: vocab_fn, gen_fn, *a = task; task = (vocab_fn, g2c(gen_fn), *a)
    counts = [9, 1]; i = 0
    while len(counts) > 1 and (len(counts) < cxt_len or counts[-1] == 1 or counts[0] > counts[-1] * 3):
        vocabs, examples = make_examples(task, nrows=nrows, cxt_len=cxt_len)
        answer_indices = [get_answer_index(e) for e in examples]
        ind_counts = Counter(answer_indices).most_common()
        counts = [v for k, v in ind_counts]
        i += 1; assert i < 10, str(ind_counts)
    if cxt_len > 1 and plot:
        print(Counter(answer_indices).most_common())
        label_probs = F.one_hot(torch.LongTensor(answer_indices))
        _ = plt.figure(figsize=(10, 0.7))
        _ = sns.heatmap(label_probs.T, cbar=False); plt.show()
    examples, text, bos_token = make_input_str(task, vocabs, examples, abstract=abstract, do_swap_qa=do_swap_qa)

    if verbose: print(text)
    return examples, text, bos_token

def task2str(task):
    vocab_fn, gen_fn, *_ = task
    return f"{fn2str(gen_fn)}({', '.join(str(v) for v in vocab_fn())})"

def is_compatible_gen_args(task, args):
    vocab_fn, gen_fn, *_ = task
    if args.get('do_swap_qa') and isinstance(gen_fn, partial) and 'query' in gen_fn.keywords: return False
    return True

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
