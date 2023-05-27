from lib2to3.pgen2 import token
import sys
import os
import json
import csv
import types
from collections import defaultdict, OrderedDict, Counter, Iterable
from functools import partial, wraps
import string
from random import choice, choices, shuffle, sample, randint, random, seed
import itertools
from dataclasses import dataclass, fields
from copy import deepcopy
import traceback
import time
import re
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn.functional as F 

from const import *
from common_utils import join_lists, list_diff, my_isinstance, lget, fn2str
from openai_utils import query_openai
from LLAMATokenizer import LLAMATokenizer
from transformers import LlamaTokenizer
# sys.path.insert(0, '/nas/xd/projects/PyFunctional')
# from functional import seq
# from functional.pipeline import Sequence

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


def types_of_characters(): return {
    'uppercase': uppercase,
    'lowercase': lowercase,
    'digit': digits,
}

# polygons = ['triangle', 'quadrangle', 'pentagon', 'hexagon', 'heptagon', 'octagon', 'nonagon', 'decagon',]# 'undecagon', 'dodecagon']
times_of_day = ['dawn', 'morning', 'noon', 'afternoon', 'evening', 'night',]# 'midnight']
clock_of_day = [f"{i} o'clock" for i in range (1, 13)]
years = [f'{i}' for i in range(2010, 2020)]
days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']#, 'Sunday']
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
temporal_posets = [clock_of_day, days_of_week, months, seasons, years]
def temporal_poset(): return choice(temporal_posets)
temporal_words = join_lists(temporal_posets)

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
    'animal': ['duck', 'goose', 'dog', 'lion', 'cow', 'donkey', 'horse', 'sheep', 'goat', 'tiger', 'cat', 'pig', 
            'monkey', 'rabbit', 'elephant', 'wolf', 'deer', 'fox', 'gorilla', 'squirrel', 'mouse'], # 'chicken', 'bear', 'zebra', 'giraffe', 'kangaroo', 21-5, 15-8
    'fruit': ['apple', 'banana', 'pear', 'grapes', 'cherries', 'orange', 'peach', 'plum', 'lemon', 'mango', 'blackberries',
            'blueberries', 'strawberries', 'durian', 'papaya', 'watermelon', 'pineapple', 'kiwi', 'apricot', 'lime'], # may be food too?
    # 'vegetable': ['spinach', 'broccoli', 'lettuce', 'cabbage', 'tomato'],
    'drink': ['tea', 'coffee', 'beer', 'wine', 'whiskey', 'vodka', 'soda', 'juice', 'cocktail'],  # some as alcohol, 21-5, 15-8
    'food': ['hamburger', 'burger', 'bread', 'meat', 'pizza', 'cake', 'steak', 'spaghetti',
            # 'biscuits', 'spaghetti', 'chips', 'peanuts', 'nuts', 'pork', 'beef', 'mutton'
            ],  # last three as meat， 21-5， 15-8
    'weapon': ['gun', 'handgun', 'shotgun', 'rifle',  'pistol', 'revolver', 'grenade', 'cannon'], #'bomb', 'dagger', 'sword',], # 21-5, 15-8, though latter prefers firearm
    'color': ['white', 'black', 'red', 'yellow', 'blue', 'green', 'purple', 'pink', 'gray'],  # 15-8
    'insect': ['mosquito', 'beetle', 'bee'], #'spider', 'ant', 'wasp', 'butterfly'],  # , 'fly'
    # 'flower': ['rose', 'tulip', 'lily', 'daisy', 'sunflower'],
    'vehicle': ['car', 'jeep', 'bus', 'taxi', 'motorcycle'],# 'tractor', 'airplane', 'ship', 'bicycle', 'truck', 'train', 'motorbike', 'helicopter', 'carriage', 
                # 'subway', 'van', 'boat'],  # transportation
    # 'furniture': ['sofa', 'couch'], #'desk', 'chair', 'table', 'bed', 'bookshelf'],# 'closet', 'wardrobe'],
    # 'tool': ['hammer', 'spanner', 'awl', 'scissors', 'saw', 'shovel', 'screwdriver', 'wrench', 'drill', 'pliers'], #, 'axe' should be weapon?
    'clothing': ['shirt', 'T-shirt', 'jeans', 'jacket', 'pants', 'trousers', 'shoes', 'sweater', 'jersey', 'underwear', 'costume', 'uniform'],#'dress', 'coat', 'socks', 'hat', 'tie', 'skirt', ],
    # 'appliance': ['microwave', 'fridge', 'washer', 'dryer', 'washing machine'],  #, 'oven'
    # 'fish': [],
    # 'country': [],
    # 'language': [],
    # 'temperature': [],
    # 'age': [],
    # 'plant': ['tree', 'grass', 'bush', 'weed', 'vine'],
    'electronics': ['laptop', 'iPad', 'phone', 'smartphone'], #'computer', 'television', 'camera', 'printer'],
    'ball': ['football', 'basketball', 'baseball'],# 'volleyball'],  # 'sport or ball?
    'musical instrument': ['piano', 'violin', 'guitar'],
    # 'utensil': ['spoon', 'fork', 'knife', 'plate', 'cup', 'bowl', 'pot'],
    # 'stationery': ['pen', 'pencil', 'paper', 'eraser', 'notebook', 'book', 'ruler', 'ink', 'stapler', 'rubber'],
}

def things(): return {thing: [thing] for thing in join_lists(types_of_things().values())}

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

# def capabilities_of_things():
#     d = defaultdict(list)
#     for thing, cap in _capabilities_of_things: d[cap].append(thing)
#     return d

def capabilities_of_things(): return {
    'kill': ['dagger', 'knife', 'gun'],
    'cook': ['oven', 'pot', 'pan'],
    'write': ['pen', 'pencil', 'chalk', 'biro'],
    'fly': ['plane', 'glider', 'helicopter'],
    'play': ['piano', 'violin', 'guitar'],
    'drive': ['car', 'truck', 'jeep'],
    'ride': ['bicycle', 'motorcycle', 'horse'],
    'communicate': ['phone', 'telephone', 'telegraph', 'radio'], # internet, email
    'clean': ['broom', 'mop', 'vacuum cleaner'],
    'paint': ['brush', 'palette', 'roller', 'spray'],
    'swim': ['swimsuit', 'goggles', 'swim fins'],
    'calculate': ['computer', 'calculator', 'abacus'],
}

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

def countries_of_cities(): return {
    'China': ['Beijing', 'Shanghai', 'Guangzhou'],
    'Japan': ['Tokyo', 'Osaka', 'Kyoto'],
    'the United Kingdom': ['London', 'Manchester', 'Birmingham'],  # England
    'the United States': ['Washington, D.C', 'New York', 'Los Angeles'],
    'Canada': ['Ottawa', 'Toronto', 'Vancouver'],
    'Australia': ['Canberra', 'Sydney', 'Brisbane'],
    'France': ['Paris', 'Marseille', 'Lyon'],
    'Italy': ['Rome', 'Milan', 'Florence', 'Venice'],
    'German': ['Berlin', 'Hamburg', 'Munich'],
    'Spain': ['Madrid', 'Barcelona', 'Valencia'],
    'Switzerland': ['Bern', 'Zurich', 'Geneva'],
    'Brazil': ['Brasília', 'Sao Paulo', 'Rio de Janeiro'],
    'India': ['New Delhi', 'Mumbai', 'Bangalore'],
    'Thailand': ['Bangkok', 'Chiang Mai', 'Pattaya'],
    'South Korea': ['Seoul', 'Busan', 'Incheon'],
    'Russia': ['Moscow', 'Saint Petersburg', 'Novosibirsk'],  # or St. Petersburg
    # 'Turkey': ['Ankara', 'Istanbul', 'Izmir'],
    # 'Argentina': ['Buenos Aires', 'Cordoba', 'Rosario'],
    # 'Mexico': ['Mexico City', 'Guadalajara', 'Monterrey'],
    # 'Egypt': ['Cairo', 'Alexandria'],
    # 'Portugal': ['Lisbon', 'Porto'],
}

def city2resident():
    if not hasattr(city2resident, 'demonyms'):
        # https://github.com/knowitall/chunkedextractor/blob/master/src/main/resources/edu/knowitall/chunkedextractor/demonyms.csv
        city2resident.demonyms = {country: resident for resident, country in csv.reader(open('demonyms.csv'))}
        city2resident.demonyms.update({'the United States': 'American', 'the United Kingdom': 'British', 'England': 'English'})
    return {capital: city2resident.demonyms[country.replace('the ', '')] for country, capital in _country2capital}

from child_frames import adjs
def a_(noun):  # prepend indefinite article a/an if possible
    if noun[0].isupper() or noun in temporal_words + adjs:
        return noun
    
    d = {'apple':  'an apple',  'chip': 'chips',  'coffee': 'coffee',  'biscuit': 'biscuits', 'dog': 'a dog', 'tea': 'tea'}
    if noun in d: return  d[noun]

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
        # if not text.split()[-1].startswith(noun[:2]): # e.g. 'red' -> 'a red apple'
        #     text = noun  # print(f'{noun} -> {text}. Skip abnormal wrap')
        # if not text.split()[-1].startswith(noun): # e.g. 'blackberry' -> 'blackberries'
        #     text = noun; print(f'{noun} -> {text}. Skip abnormal wrap')
        if noun not in text or text.split(noun)[-1].startswith(' '):
            text = noun#; print(f'{noun} -> {text}. Skip abnormal wrap')
        return text
    return extract_fn(query_openai(prompt_fn(noun), 'text-davinci-003')) # by lxy

wrap_noun = a_

def strip_a(text):
    if text.startswith('a ') or text.startswith('an '):
        text = re.sub(r"^a ", "", text); text = re.sub(r"^an ", "", text)
    return text

def the_(noun, uppercase=True):
    if noun.lower() in ['who', 'which']:   # in swap_qa
        return capitalize(noun) if uppercase else noun
    if noun[0].isupper(): return noun  # proper noun
    the = 'The' if uppercase else 'the'
    return the + ' ' + strip_a(noun)

def _s(noun):  # to plural form if possible
    prompt_fn = lambda s: \
f'''apple: He likes apples.
tea: He likes tea.
red: He likes red.
trousers: He likes trousers.
dog: He likes dogs.
{s}: He likes'''
# f'''Change words into plural forms if possible.
# apple: apples
# chip: chips
# coffee: coffee
# biscuit: biscuits
# red: red
# trousers: trousers
# dog: dogs
# tea: tea
# {s}:'''
    def extract_fn(text):
        text = text.strip()
        if text.endswith('.'): text = text[:-1]
        # if text.startswith('a ') or text.startswith('an '):
        #     print(f'In nouns: {noun} -> {text}')
        #     text = re.sub(r"^a ", "", text); text = re.sub(r"^an ", "", text)
        if not text.startswith(noun[:1]): print(f'In nouns: {noun} -> {text}')
        return text
    noun = strip_a(noun)
    d = {'drink': 'drinks'}
    if noun in d: return d[noun]
    return extract_fn(query_openai(prompt_fn(noun), 'text-davinci-003'))  # better than text-davinci-002

def _be(noun): # by xd
    prompt_fn = lambda s: \
f'''The apple is here.
The coffee is here.
The shoes are here.
The red is here.
The blueberries are here.
{s} '''
    def extract_fn(text):
        text = text.strip()
        if text.endswith('.'): text = text[:-1]
        if text.split(' ')[0] not in ('is', 'are'): return 'is'
        return text.split(' ')[0]
    prompt = prompt_fn('')
    match = re.search(noun + ' ', prompt)
    if match is not None: return noun + ' ' + prompt[match.end():].split(' ', 1)[0]
    dictss = {'The bread': 'The bread is', 'The cake': 'The cake is', 'The pizza': 'The pizza is', 'The cherries':'The cherries are'}
    if noun in dictss: return dictss[noun] 
    return noun + ' ' + extract_fn(query_openai(prompt_fn(noun), 'text-davinci-003'))

def wrap_noun_to_french(noun):
    prompt_fn = lambda s: \
f'''dog: Martin a un chien.
apple: Martin a une pomme.
tea: Martin a du thé.
{s}: Martin a'''
    def extract_fn(text):
        assert text.endswith('.'); text = text[:-1]
        return text.strip()  # strip leading spaces
    return extract_fn(query_openai(prompt_fn(noun), 'text-davinci-003')) if noun != 'tea' else 'du thé'

def prep_(noun):
    if noun in clock_of_day: return 'at ' + noun
    if noun in days_of_week: return 'on ' + noun
    if noun in months: return 'in ' + noun
    if noun in seasons: return 'in ' + noun
    if noun in years: return 'in ' + noun
    assert False
    prompt_fn = lambda s: \
f'''summer: He arrived in the summer.
Friday: He arrived on Friday.
morning: He arrived in the morning.
February: He arrived in February.
5 o'clock: He arrived at 5 o'clock.
2020: He arrived in 2020.
{s}: He arrived'''
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

def wrap(fn, k_wrapper=None, v_wrapper=None):
    def wrapped_fn():
        return {k_wrapper(k) if k_wrapper else k: [v_wrapper(v) if v_wrapper else v for v in values]
            for k, values in fn().items()}
    wrapped_fn.__name__ = '.'.join(f.__name__ for f in [k_wrapper, fn, v_wrapper] if f)
    return wrapped_fn

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
    
synonym_dict = {
    'has': ['owns', 'possesses'], 'have': ['own', 'possesse'],
    'wants to go to': ['wants to visit', 'longs for', 'yearns for'], 'want to go to': ['want to visit', 'long for', 'yearn for'],
    'arrived': ['appeared', 'showed up'], 'arrive': ['appear', 'show up'],
}
synonym_dict = {k: [k] + v for k, v in synonym_dict.items()}

def sampled_synonym_dict(): return {k: choice(v) for k, v in synonym_dict.items()}

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
        self.neg_rel = None
        self.x_f = None
        self.y_f = None
        self.skip_inv_f = False

    def f(self, x): return self._dict.get(x, [])
    def inv_f(self, x): return self._inv_dict.get(x, [])
    def dom(self, xs=None): return list(self._dict.keys())
    def codom(self, ys=None):
        elems = join_lists(self._dict.values()) if self.name != 'sibling' else list(self._dict.keys())
        if self.name in ['parent', 'opposite']: elems = list(set(elems))
        else: assert len(elems) == len(set(elems)), f'{self.name} {len(elems)} != {len(set(elems))}'
        return elems

    def b(self, x0, x1): return x1 in self._dict.get(x0, [])

    def __str__(self):
        s = self.name if not self.skip_inv_f else 'equal'
        def attr2str(name):
            value = getattr(self, name)
            return f'{name}={value.__name__}' if value != True else name
        attr_str = ','.join(attr2str(name) for name in ['x_f', 'y_f']
                                        if getattr(self, name) not in [None, False])
        if attr_str != '': s += f'[{attr_str}]'
        return s

class NegativeRelation(Relation):
    def __init__(self, rel, set_obj):
        self.rel = self.neg_rel = rel
        rel.neg_rel = self
        self.name = 'neg_' + rel.name if not rel.name.startswith('neg_') else rel.name[4:]
        for name in ['x_f', 'y_f', 'skip_inv_f']: setattr(self, name, getattr(self.rel, name))
        if self.name == 'neg_equal' and hasattr(set_obj, 'sibling'):  # set_obj is a TreeSet or SymSet
            self.sibling = set_obj.sibling  # used in distractive_sample

    def f(self, x): return list_diff(self.rel.codom(), self.rel.f(x))
    def inv_f(self, x): return list_diff(self.rel.dom(), self.rel.inv_f(x))
    def dom(self, xs=None): return self.rel.dom()
    def codom(self, ys=None): return self.rel.codom()
    def b(self, x0, x1): return not self.rel.b(x0, x1)

class Set(object):
    def __init__(self, data, rel_names):
        self.data = data
        self.rel_names = rel_names
        for rel_name in self.rel_names:
            setattr(self, rel_name, Relation(name=rel_name, _dict=defaultdict(list)))

    def use(self, rel_names, x_f=None, y_f=None, skip_inv_f=False):
        if isinstance(rel_names, str): rel_names = [rel_names]
        self.relations = [getattr(self, rel_name) if not rel_name.startswith('neg_') else 
            NegativeRelation(getattr(self, rel_name.replace('neg_', '')), self) for rel_name in rel_names]
        for rel in self.relations[:1]:  # TODO: check compatibility with NegativeRelation
            rel.x_f, rel.y_f, rel.skip_inv_f = x_f, y_f, skip_inv_f
        return self

    def negate_used(self):
        self.relations = [NegativeRelation(rel, self) for rel in self.relations]
        return self

    def __str__(self):
        return f"{self.data.__name__}.{self.__class__.__name__}.{'|'.join(str(rel) for rel in self.relations)}"

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
        data = self._data = data()
        for rel_name, d in zip(self.rel_names, [{data[i]: [data[i - 1]] for i in range(1, len(data))},
                                                {data[i]: [data[i + 1]] for i in range(0, len(data) - 1)},
                                                {data[i]: [data[i]] for i in range(1, len(data) - 1)}]):
            setattr(self, rel_name, Relation(name=rel_name, _dict=d))
        self.prev._inv_dict, self.next._inv_dict = self.next._dict, self.prev._dict
        self.equal._inv_dict = self.equal._dict
        self.prev.inv_rel, self.next.inv_rel = self.next, self.prev
        self.equal.inv_rel = self.equal

class SymSet(Set):
    def __init__(self, data):
        super().__init__(data, ['similar', 'opposite', 'sibling', 'equal'])
        data = data()
        for pair in data:
            for similars, opposites in [(pair[0], pair[1]), (pair[1], pair[0])]:
                for e in similars:
                    self.equal._dict[e] = [e]
                    if len(similars) > 1: self.similar._dict[e] = list_diff(similars, [e])
                    self.opposite._dict[e] = opposites[:1]
                    self.sibling._dict[e] = list_diff(similars, [e]) + opposites  # used by neg_equal
        self.opposite._inv_dict, self.similar._inv_dict = self.opposite._dict, self.similar._dict
        self.equal._inv_dict = self.equal._dict
        self.sibling._inv_dict = self.sibling._dict
        self.similar.inv_rel, self.opposite.inv_rel = self.similar, self.opposite
        self.equal.inv_rel = self.equal
        self.sibling.inv_rel = self.sibling
        
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
        super().__init__(data, ['child', 'parent', 'sibling', 'equal'])
        data = data()
        for parent, children in data.items():
            self.child._dict[parent] = children
            # self.equal._dict[parent] = [parent]
            for child in children:
                self.parent._dict[child] = [parent]
                self.equal._dict[child] = [child]
                self.sibling._dict[child] = list_diff(children, [child])
        self.child._inv_dict, self.parent._inv_dict = self.parent._dict, self.child._dict
        self.sibling._inv_dict = self.sibling._dict
        self.equal._inv_dict = self.equal._dict
        self.child.inv_rel, self.parent.inv_rel = self.parent, self.child
        self.sibling.inv_rel = self.sibling
        self.equal.inv_rel = self.equal

def fork_vocab(vocab,  rel_names_list):
    return [v.use(rel_names) for v, rel_names in zip([vocab, deepcopy(vocab)], rel_names_list)]

def enumerate_sample(cxt_len, rel):
    return list(range(cxt_len)), sample(rel.codom(), cxt_len)

def grouped_sample(cxt_len, rel, n_groups=2, reverse=False, min_group_size=None, max_group_size=None,
                                                            min_group_count=None, max_group_count=None):
    if min_group_size and not max_group_size: min_group_count, max_group_count = 1, None
    elif not min_group_size and max_group_size: min_group_count, max_group_count = None, 1
    group_sizes = []
    i = 0
    while (not group_sizes or (min_group_size and min(group_sizes) != min_group_size)
                        or (max_group_size and max(group_sizes) != max_group_size)
                        or (min_group_count and group_sizes.count(min(group_sizes)) != min_group_count)
                        or (max_group_count and group_sizes.count(max(group_sizes)) != max_group_count)):
        cut_points = sorted(sample(range(1, cxt_len - 1), n_groups - 1))
        group_sizes = [stop - start for start, stop in zip([0] + cut_points, cut_points + [cxt_len])]
        group_sizes = sorted(group_sizes, reverse=reverse)
        # print('In grouped_sample:', group_sizes)  # debug
        i += 1; assert i <= 10, str(group_sizes)
    groups = sample(rel.dom(), n_groups)
    cxt = join_lists([list(zip([size] * size, sample(rel.f(group), size) if rel.name != 'equal' else [group] * size))
                    for group, size in zip(groups, group_sizes)])
    return tuple(map(list, zip(*cxt)))

def swap(l, dst, src=0):
    if dst != src: l[dst], l[src] = l[src], l[dst]
    return l

def distractive_sample(cxt_len, rel, ans_i=0):
    query = choice(rel.dom())
    siblings = rel.sibling.f(query) if rel.name == 'neg_equal' and hasattr(rel, 'sibling') else []
    ans = choice(list_diff(rel.f(query), siblings))
    distractors = list_diff(rel.codom(), rel.f(query) + ([query] if rel.name == 'sibling' else []))
    k = cxt_len - 1
    # neg_xxx or parent rel may have only one distractor
    assert len(distractors) >= k or len(distractors) == 1, \
        f'{rel.name}, query = {query}, f(query) = {rel.f(query)}, distractors = {distractors}'
    distractors = sample(distractors, k) if len(distractors) >= k else distractors * k
    # TODO: rel.inv_f(x)[0] -> choice(rel.inv_f(x)). check not equivalent for sibling
    distractors0 = [choice(rel.inv_f(x)) for x in distractors]
    candidates = [[query] + distractors0, [ans] + distractors]
    if rel.skip_inv_f and rel.x_f is None: rel.x_f = lambda x: x
    if rel.x_f: candidates[0] = [rel.x_f(c) for c in candidates[int(rel.skip_inv_f)]]
    if rel.y_f: candidates[1] = [rel.y_f(c) for c in candidates[1]]
    return tuple([swap(l, ans_i) for l in candidates])

def MlM_gen(vocabs, cxt_len=3, cxt_sample_fn=None, query=None):
    rels = [s.relations for s in vocabs]
    candidates = OrderedDict()
    fixed_query = query is not None
    has_local_hop = vocabs[0].data != vocabs[1].data
    position_relevant = getattr(cxt_sample_fn, '__name__', None) == 'enumerate_sample'
    
    hop = 0; rel = rels[hop][0]
    candidates[hop - 1], candidates[hop] =  distractive_sample(cxt_len, rel) \
        if not fixed_query else cxt_sample_fn(cxt_len, rel)
    # if not fixed_query: query = candidates[hop - 1][0]
    # elif not position_relevant: assert query == candidates[hop - 1][0], f'{query} != {candidates[hop - 1][0]}'

    hop = 1; rel = rels[hop][0]
    candidates[hop], candidates[hop + 1] = distractive_sample(cxt_len, rel)[::-1] if has_local_hop \
        else (candidates[hop - 1].copy(), [choice(rel.inv_f(x)) for x in candidates[hop - 1]])

    tuples = list(zip(*candidates.values()))
    i = 0 if query is None else list(candidates.values())[0].index(query)
    if query is not None: assert tuples[i][0] == query, f'{tuples[i]}[0] != {query}'
    query, *ans_chain = tuples[i]; ans_chain = tuple(ans_chain)
    if not position_relevant: shuffle(tuples)
    cxt = [t[int(not fixed_query):3] for t in tuples]
    candidates = tuple(list(c) for c in zip(*tuples))

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

def _str(l, vocab=None, sep=' '):
    if l is None: return ''
    if isinstance(l, str) or not isinstance(l, Iterable): l = [l]
    # l = [e for e in l if not my_isinstance(e, Sequence)] #type(e).__name__ != 'Sequence']
    if isinstance(l, (dict, OrderedDict)): l = [f'{k}: {v}' for k, v in l.items()]
    return sep.join(str(i) for i in l)

def options2str(options): return '[' + ' | '.join(options) + ']'  # ' or '.join(options) + '?'

def make_examples(task, nrows=4, vocab_for_each_row=False, **kwargs):
    vocab_fn, example_gen_fn = task[:2]
    vocabs, examples = [], []
    qa_set = set() # for dedup
    if any(v.__class__.__name__ == 'PoSet' for v in vocab_fn()):
        vocab_for_each_row = True
    if not vocab_for_each_row: vocab = vocab_fn()
    for i in range(nrows * 2):
        if vocab_for_each_row: vocab = vocab_fn()
        cxt, query, candidates, ans_chain, *a = example_gen_fn(vocab, **kwargs)
        if isinstance(query, list): query = tuple(query)
        if (tuple(cxt), query, ans_chain) not in qa_set:
            qa_set.add((tuple(cxt), query, ans_chain))
            vocabs.append(vocab)
            examples.append([cxt, query, candidates, ans_chain, *a])
        if len(examples) == nrows: break
    return vocabs, examples

def _item2str(item, vocab=None): #, reverse=False):
    return [f'{item[0]} {item[1]}', f'{item[1]} {item[0]}'] if isinstance(item, tuple) else f'{item}'

def _cxt2str(cxt, vocab=None, prefix='< ', suffix=' >.', sep=' ', item2str=_item2str, rev_item2str=False):
    def try_wrap(s):
        return [s] if type(s) == str else s
    return prefix + sep.join([try_wrap(item2str(item, vocab))[int(rev_item2str)] for item in cxt]) + suffix

@dataclass
class Ranges:
    bos: tuple = None
    ans: tuple = None
    ans0: tuple = None
    query: tuple = None
    tgt: tuple = None
    sep: tuple = None
    ans0s: list = None
    example: tuple = None

@dataclass
class IOIRanges:   # wab
    bos: tuple = None
    ans: tuple = None
    ans0: tuple = None
    s1: tuple = None
    s2: tuple = None
    ans0s: list = None
    example: tuple = None

# adapted from find_token_range in https://github.com/kmeng01/rome/blob/main/experiments/causal_trace.py
def locate(tokens, substring, return_last=False, return_all=False):
    if substring is None: return None
    substring = substring.lower()
    substring = strip_a(substring)
    tokens = [t.lower() for t in tokens]
    whole_string = "".join(t for t in tokens)
    assert substring in whole_string, f'{tokens}\n{substring} not in {whole_string}'
    if substring.strip() in ['->', '?']:
        char_locations = [whole_string.index(substring), whole_string.rindex(substring)]
    else:
        pattern = r"\b%s(?:s|es)?\b" if not substring.startswith(" ") else r"%s(?:s|es)?\b"
        try: matches = list(re.finditer(pattern % substring, whole_string))
        except Exception: print(f'sub = {substring}, whole = {whole_string}'); raise
        assert len(matches) > 0, f'{tokens}\n{substring} not match {whole_string}'
        char_locations = [m.span()[0] for m in matches]
    if not return_all: char_locations = [char_locations[-int(return_last)]]
    ranges = []
    for char_loc in char_locations:
        loc = 0; tok_start, tok_end = None, None
        for i, t in enumerate(tokens):
            loc += len(t)
            _t = t[1:] if t.startswith(' ') else t
            forms = [substring, substring + 's', substring + 'es']
            if tok_start is None and loc > char_loc:
                assert any(s.find(_t) in [0, 1] for s in forms), \
                    f'{whole_string}\n{tokens}\n{substring} not startswith {_t} at {i}. loc = {loc}, char_loc = {char_loc}'
                tok_start = i
            if tok_end is None and loc >= char_loc + len(substring):
                assert any(s.endswith(_t) for s in forms), \
                    f'{whole_string}\n{tokens}\n{substring} not endswith {_t} at {i}'
                tok_end = i + 1
                break
        assert tok_start is not None and tok_end is not None, f'{tok_start}, {tok_end}'
        if not return_all: return (tok_start, tok_end)
        ranges.append((tok_start, tok_end))
    return ranges

def example2ranges(example, tokens, bos_token, trimmed=False):
    if 'IO' in example:  # ioi task
        e = example
        ranges = IOIRanges(
            bos = (e['end'].item(), e['end'].item()+1),
            ans = (e['end'].item()+1, e['end'].item()+2),
            ans0 = (e['IO'].item(), e['IO'].item()+1),  # io == ans0
            s1 = (e['S'].item(), e['S'].item()+1),
            s2 = (e['S2'].item(), e['S2'].item()+1),
        )
        ranges.ans0s = tuple(map(np.array, zip(ranges.ans0, ranges.s1)))  # XD: remove ranges.s2
        ranges.example = (0, ranges.ans[-1])  # XD
        return ranges

    cxt, query, candidates, (tgt, *_, ans0, ans), *cls = example
    if trimmed: return Ranges(bos = locate(tokens, bos_token, return_last=True))
    ranges = Ranges(
        bos = locate(tokens, bos_token, return_last=True),
        ans = locate(tokens, ans, return_last=True),
        ans0 = locate(tokens, ans0),
        query = locate(tokens, query, return_last=True),
        tgt = locate(tokens, tgt),
        example = (0, len(tokens))
    )
    if candidates is not None:
        max_i = ranges.query[0] if ranges.query is not None else ranges.ans[0]
        ranges.ans0s = tuple(map(np.array, zip(*filter(lambda x: x[0] < max_i, join_lists(
            [locate(tokens, a0, return_all=True) for a0 in candidates[-2]], dedup=True)))))
    if ranges.tgt is not None and '.' in tokens[ranges.tgt[1]:]:  # TODO: debug
        sep_i = tokens.index('.', ranges.tgt[1])
        ranges.sep = (sep_i, sep_i + 1)
    return ranges

def move_ranges(r, offset): 
    for field in fields(r):
        name = field.name; pair = getattr(r, name)
        if pair is not None: setattr(r, name, tuple([i + offset for i in pair]))
    return r

def locate_ranges(examples, example_strs, tokenizer, input_ids, bos_token):
    assert len(examples) == len(example_strs)
    ranges = []
    use_llama_tokenizer = my_isinstance(tokenizer, (LLAMATokenizer, LlamaTokenizer))
    newline_token = '<0x0A>' if use_llama_tokenizer else '\n' #tokenizer.tokenize('\n')[0]  # 'Ċ' \n两者表示方法不同
    if use_llama_tokenizer:  # add by lxy
        # tokenizer.decode will strip leading '__'
        all_tokens_llama = [tokenizer.convert_ids_to_tokens(id).replace('▁', ' ') for id in input_ids]
        assert all_tokens_llama[0] == tokenizer.bos_token, str(all_tokens_llama)
        all_tokens = [tokenizer.bos_token]
        assert all_tokens_llama[1].startswith(' '), all_tokens_llama[1]
        all_tokens_llama[1] = all_tokens_llama[1][1:]
        all_tokens_llama = all_tokens_llama[1:]  # treat leading bos as prefix_token and remove it from all_tokens_llama
        # split all_tokens_llama using newline_token as delimiter
        # https://stackoverflow.com/questions/15357830/splitting-a-list-based-on-a-delimiter-word
        sep_tokens = [list(y) for x, y in itertools.groupby(all_tokens_llama, lambda z: z == newline_token) if not x]
        assert len(sep_tokens) == len(example_strs)
    else:
        all_tokens = [newline_token] if tokenizer.decode(input_ids[0]) == newline_token else []
    for i, (e, e_str) in enumerate(zip(examples, example_strs)):
        # tokens = tokenizer.tokenize(e_str)  # can not work with locate
        tokens = sep_tokens[i] if use_llama_tokenizer else \
            [tokenizer.decode(id) for id in tokenizer.encode(e_str)] # lxy: LLAMATokenizer decode不出单词前面的空格
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
    input_ids = tokenizer.encode(text, return_tensors='pt')
    example_strs = text.strip('\n').split('\n')  # strip the trailing '\n'
    ranges = locate_ranges(examples, example_strs, tokenizer, input_ids[0].tolist(), bos_token)
    # by lxy: when bos is tokenized into multiple tokens, e.g. 'likes' -> ['__lik', 'es'] in LLaMA, use last token's index
    bos_indices = [r.bos[-1] - 1 for r in ranges]  # [r.bos[0] for r in ranges]
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

def capitalize(s):  # different from str.capitalize() in more than one way!
    if s.startswith(' '): return ' ' + capitalize(s[1:])
    return s[0].upper() + s[1:] if s else ''

def rsplit_bos(s):
    if s.endswith("'s"): return "'s"
    if s.endswith("?"): return "?"
    return ' ' + s.split()[-1]

def post_compose(fn, fn2):
    def new_fn(*args, **kwargs):
        return fn2(fn(*args, **kwargs))
    return new_fn

def multi_replace(s, pairs):
    for old, new in pairs.items():
        if old in s: s = re.sub(r"\b%s\b" % old, new, s)
    return s

def make_input_str(task, vocabs, examples, rev_item2str=False, abstract=False, options_position=None, tokenizer = None):
    # Randomized transformations here are per input basis, i.e. each example in an input are the same,
    # while each input in a task's batch may be different. It is finer-grained than transform_task which are per task basis.
    # Hierarchy: task >= batch > input > example
    cxt, *_ = examples[0]
    cxt_len = len(cxt)
    if abstract:
        prefix="< "; suffix=" >."; query2str = _str
        if cxt_len == 1:
            examples = [(cxt, None, None, (None, ans0, ans), *cls)
                for cxt, query, options, (tgt, ans0, ans), *cls in examples]
            prefix=""; suffix=""; query2str = lambda q, _: ''
        sep = choice([', ', ' ']) if isinstance(cxt[0], str) else ' '
        i2s = choice([lambda i: f'( {i[0]}, {i[1]} )', lambda i: f'( {i[0]} {i[1]} )',
                    lambda i: f'{i[0]}, {i[1]}.', lambda i: f'{i[0]} {i[1]}.'])
        def item2str(item, vocab=None):
            return [i2s(i) for i in [item, item[::-1]]] if isinstance(item, tuple) else f'{item}'
        cxt2str = partial(_cxt2str, prefix=prefix, suffix=suffix, sep=sep, item2str=item2str)
        bos_token, ans2str = abstract_bos_token, _str
    else:
        cxt2str, query2str, bos_token, ans2str = [lget(task, i, '' if i == 4 else _str) for i in range(2, 6)]
        query2str = post_compose(query2str, partial(multi_replace, pairs=sampled_synonym_dict()))
    def example2str(vocab, example):
        cxt, query, candidates, (*_, ans), *cls = example
        strs = [cxt2str(cxt, vocab, rev_item2str=rev_item2str), capitalize(query2str(query, vocab))]
        if options_position is not None: strs.insert(options_position, options2str([c[-1] for c in candidates]))
        s = ' '.join(s for s in strs if s != '') + bos_token + ' ' + ans2str(ans)
        _bos_token = bos_token
        if bos_token == '': query_str = strs[1]; _bos_token = rsplit_bos(query_str)
        if len(cls) > 0: _bos_token = '?'; s += _bos_token + ' ' + _str(cls[0]) # g2c
        return s, _bos_token
    example_strs, bos_tokens = zip(*[example2str(v, e) for v, e in zip(vocabs, examples)])
    text = '\n '.join(example_strs) + '\n' if isinstance(tokenizer, (LLAMATokenizer, LlamaTokenizer)) else \
        '\n' + '\n'.join(example_strs) + '\n'  # prepend '\n' to act as bos for tokenizer without bos
    return examples, text, bos_tokens

def get_answer_index(example):
    cxt, query, cands, (*_, ans), *cls = example
    return cands[-1].index(ans)

class InvalidTransException(Exception): pass

def choose_rels(task, rel_indices):
    vocab_fn, gen_fn, cxt2str, query2str, *a = task
    vocabs = vocab_fn()
    for hop, rel_i in enumerate(rel_indices):
        if rel_i >= len(vocabs[hop].relations): return None

    def new_vocab_fn():
        vocabs = vocab_fn()
        for hop, rel_i in enumerate(rel_indices):
            vocabs[hop].relations = vocabs[hop].relations[rel_i: rel_i + 1]
        return vocabs
    
    task = new_vocab_fn, gen_fn, cxt2str, query2str, *a
    return task

def decorate_rel(task, hop, kwargs):
    vocab_fn, gen_fn, cxt2str, query2str, *a = task

    def new_vocab_fn():
        vocabs = vocab_fn()
        rel = vocabs[hop].relations[0]
        for k, v in kwargs.items(): setattr(rel, k, v)
        return vocabs

    task = new_vocab_fn, gen_fn, cxt2str, query2str, *a
    return task

def get_wh_and_the(vocab):
    data_name, rel_name = vocab.data.__name__, vocab.relations[0].name
    wh = 'who' if data_name in ['persons', 'genders_of_persons'] else 'which'
    the = ' the' if data_name == 'genders_of_persons' and rel_name == 'child' or \
        data_name == 'capabilities_of_things' and rel_name != 'child' or \
        data_name == 'types_of_things' and rel_name != 'child' else ''
    return wh, the

def verbalize_relation(vocab):
    data_name, rel_name = vocab.data.__name__, vocab.relations[0].name
    _rel_name = rel_name.split('neg_')[-1]
    r2v = {'genders_of_persons': ('', 'one'),
        'types_of_things': (' a kind of', 'one'),
        'capabilities_of_things': (' the thing that can', 'one'),
        'countries_of_cities': (' the city in', 'city'), 
    }
    if _rel_name in ['prev', 'next']:
        temporal_word = {tuple(clock_of_day): 'time', tuple(days_of_week): 'day', tuple(months):'month',
            tuple(seasons): 'season', tuple(years): 'year'}[tuple(vocab._data)]
    if _rel_name == 'child': rel_str = r2v[data_name][0]
    elif _rel_name == 'sibling': rel_str = f' the {r2v[data_name][1]} like'
    elif _rel_name == 'prev': rel_str = f' the {temporal_word} just before'
    elif _rel_name == 'next': rel_str = f' the {temporal_word} just after'
    elif _rel_name == 'opposite': rel_str = ' the opposite of'
    else: rel_str = ''
    return rel_str

def refine_query2str(task, do_swap_qa=False):
    vocab_fn, gen_fn, cxt2str, query2str, *a = task
    if query2str is None: return task
    rel_names = [vocab.relations[0].name for vocab in vocab_fn()]
    def new_query2str(q, v):
        # Tricky: new_query2str called and v passed AFTER swap_qa, so v has been swapped
        return query2str(q, v) + verbalize_relation(v[1 - do_swap_qa])
    task = (vocab_fn, gen_fn, cxt2str, new_query2str, *a)
    return task

def swap_qa(task):
    vocab_fn, gen_fn, cxt2str, query2str, *a = task
    def new_vocab_fn(): return vocab_fn()[::-1]  # would cause infinite recursion bug if use same name
    item2str = cxt2str.keywords['item2str']
    swapped_item2str = lambda i, v: item2str(i[::-1], v)
    new_cxt2str = deepcopy(cxt2str)
    new_cxt2str.keywords['item2str'] = swapped_item2str

    def new_query2str(q, v):
        wh, the = get_wh_and_the(v[1])
        return f'{query2str(wh, v)} {q}?'.replace("who's", "whose") + capitalize(the)
    task = (new_vocab_fn, gen_fn, new_cxt2str, new_query2str, *a)
    return task

def negate_sent(s):  # TODO: a more principled way of negating a sentence
    s00 = s0 = s
    n_replaced = 0
    for old, new in [
        # (" likes", " does not like"), (" owns", " does not own"), (" possesses", " does not possess"),
        (" wants ", " does not want "), #(" wanna ", " does not want to "),
        (" arrived", " did not arrive"),
        (r"\bis\b", "is not"), (r"\bhas\b", "does not have")]:
        if r"\b" in old: s = re.sub(old, new, s0)
        else: s = s0.replace(old, new)
        if s != s0:
            n_replaced += 1
            # skip 'has' if 'is' is present. e.g. 'What John has is' -> 'What John has is not'
            if old == r"\bis\b": break
        s0 = s
    assert n_replaced == 1, f'n_replaced = {n_replaced}: {s00} -> {s}'

    singular_subs, plural_subs = ['boy ', 'girl '], ['boys ', 'girls ']
    sep = r"\bnot\b" #if " not " in s else r"\bis\b"
    sep_i = list(re.finditer(sep, s))[0].span()[0]
    if any(sub in s[:sep_i] for sub in singular_subs):
        for old_sub, new_sub in zip(singular_subs, plural_subs):
            s = s.replace(old_sub, new_sub)
        s = s.replace(" does not", " do not")
    return s

def negate(task):
    vocab_fn, gen_fn, cxt2str, query2str, *a = task

    def new_vocab_fn():
        vocabs = vocab_fn()
        return [vocabs[0].negate_used(), vocabs[1]]

    new_query2str = (lambda q, v: negate_sent(query2str(q, v))) \
        if query2str is not None else None

    task = (new_vocab_fn, gen_fn, cxt2str, new_query2str, *a)
    return task
    
def remove_local_hop(task, do_rm_query, do_g2c):
    vocab_fn, gen_fn, cxt2str, query2str, *a = task
    vocabs = vocab_fn()
    assert vocabs[0].data == vocabs[1].data
    data_name = vocabs[0].data.__name__
    rel_names = [v.relations[0].name for v in vocab_fn()]
    fixed_query = isinstance(gen_fn, partial) and 'query' in gen_fn.keywords
    
    assert not rel_names[1].startswith('neg_'), rel_names[1]
    if fixed_query:
        pass  # TODO: Is there any rule for fixed_query?
    elif rel_names[0] == 'equal' or rel_names[1] in ['child', 'sibling'] and not rel_names[0].startswith('neg_'):
        raise InvalidTransException("invalid rel for rm_local_hop: " + str(rel_names))
    elif rel_names[1] in ['child', 'sibling'] and len(vocabs[1].child.dom()) == 2 and not do_rm_query:
        raise InvalidTransException(f"invalid rel for rm_local_hop: {rel_names}. len({data_name}.child.dom()) == 2")
    elif not do_rm_query and do_g2c:  # solvable without cxt
        raise InvalidTransException(f"invalid rel for rm_local_hop with g2c: {rel_names}")
    
    if cxt2str is None:
        cxt2str = partial(_cxt2str, prefix='There are ', suffix='.', sep=', ', item2str=lambda i, _: a_(i))
    
    if query2str is None:
        if not fixed_query:
            def query2str(q, v):
                wh, the = get_wh_and_the(v[1])
                rel_str = verbalize_relation(v[1]) + the
                neg_str = ' not' if rel_names[0].startswith('neg_') else ''
                return f"{wh} is{neg_str}{verbalize_relation(v[0])} {q}?" + capitalize(rel_str)
        else:
            query2str = lambda q, v: ""
    task = vocab_fn, gen_fn, cxt2str, query2str, *a
    return task

def remove_query(task):
    vocab_fn, gen_fn, cxt2str, query2str, *a = task
    vocabs = vocab_fn()
    rel_names = [v.relations[0].name for v in vocabs]
    if not rel_names[0].startswith('neg_') or rel_names[0] == 'neg_sibling':  # neg_sibling == neg_child
        raise InvalidTransException("invalid rel for rm_query" + str(rel_names))

    def new_gen_fn(*args, **kwargs):
        cxt, query, candidates, (tgt, *a, ans0, ans) = gen_fn(*args,**kwargs)
        query, candidates = None, ([None] * len(candidates[1]),) + candidates[1:]
        return cxt, query, candidates, (tgt, *a, ans0, ans)
    new_gen_fn.__name__ = f"rm_query[{fn2str(gen_fn)}]"

    def new_query2str(q, v):
        wh, the = get_wh_and_the(v[1])
        rel_str = verbalize_relation(v[1]) + the
        return f"{wh} is different?" + capitalize(rel_str)
    task = vocab_fn, new_gen_fn, cxt2str, new_query2str, *a
    return task

def _g2c(g_fn, cls_labels=['True', 'False']):
    def wrapped(*args,**kwargs):
        cxt, query, candidates, (*a, ans0, ans) = g_fn(*args,**kwargs)
        (_ans0, _ans), label = ((ans0, ans), cls_labels[0]) if random() < 0.5 else \
            (choice([(c0, c) for q, *_, c0, c in zip(*candidates) if c != ans and (query is None or q != query)]), cls_labels[1])
        return cxt, query, candidates, (*a, _ans0, _ans), label
    wrapped.__name__ = f'g2c[{fn2str(g_fn)}]'
    return wrapped

def g2c(task):
    vocab_fn, gen_fn, *a = task
    task = (vocab_fn, _g2c(gen_fn), *a)
    return task

def has_local_hop(task):
    vocab_fn, *a = task; vocabs = vocab_fn()
    return vocabs[0].data != vocabs[1].data

def transform_task(task, rel0_i=None, rel1_i=None, #replace_rel0=0, replace_rel1=0,
                rel0_kwargs=None, rel1_kwargs=None, do_swap_qa=False, do_negate=False,
                do_rm_query=False, do_g2c=False):
    try:
        if rel0_i is not None: task = choose_rels(task, [rel0_i, rel1_i])
        if task is None: return None
        if rel0_kwargs is not None: task = decorate_rel(task, 0, rel0_kwargs)
        if rel1_kwargs is not None: task = decorate_rel(task, 1, rel1_kwargs)
        task = refine_query2str(task, do_swap_qa=do_swap_qa)
        if not has_local_hop(task) and do_swap_qa: return None
        if do_swap_qa: task = swap_qa(task)
        if do_negate: task = negate(task)
        if not has_local_hop(task): task = remove_local_hop(task, do_rm_query, do_g2c)
        if do_rm_query: task = remove_query(task)
        if do_g2c: task = g2c(task)
    except InvalidTransException as e:
        trans_args = {k: v for k, v in locals().items() if k not in ['task', 'e']}
        print(f'\ntransform_task failed: {e} ({args2str(trans_args)})')
        return None
    return task

def generate(task, nrows=8, cxt_len=3, rev_item2str=False, abstract=0, tokenizer = None, max_length=512, plot=True, verbose=True):
    vocab_fn, gen_fn, cxt2str, query2str, *a = task
    position_relevant = isinstance(gen_fn, partial) and \
        'cxt_sample_fn' in gen_fn.keywords and 'query' in gen_fn.keywords and \
        gen_fn.keywords['cxt_sample_fn'].__name__ == 'enumerate_sample'

    ans_counts = [('a', nrows)]; ind_counts = [(0, 9), (1, 1)]; i = 0
    while (not position_relevant and len(ind_counts) > 1 and (len(ind_counts) < cxt_len 
                                    or ind_counts[0][1] > ind_counts[-1][1] * 3) or
            len(ans_counts) == 1 or
            len(ans_counts) > 2 and ans_counts[0][1] > nrows / 4 or
            len(ans_counts) == 2 and ans_counts[0][1] >= ans_counts[1][1] * 2):
        vocabs, examples = make_examples(task, nrows=nrows, cxt_len=cxt_len)
        # print('In generate: example =', examples[0])
        ans_counts = Counter([ans for cxt, query, cands, (*_, ans), *cls in examples]).most_common()
        answer_indices = [get_answer_index(e) for e in examples]
        ind_counts = Counter(answer_indices).most_common()
        i += 1
        assert i < 25, '\n'.join(f'{e[0]}\t{e[1]}\t{e[3]}' for e in examples[:]) + \
            '\n' + str(ind_counts) + '\n' + str(ans_counts)
    # if i > 1: print('In generate: i =', i)
    if cxt_len > 1 and plot:
        print(Counter(answer_indices).most_common())
        label_probs = F.one_hot(torch.LongTensor(answer_indices))
        _ = plt.figure(figsize=(10, 0.7))
        _ = sns.heatmap(label_probs.T, cbar=False); plt.show()
    examples, text, bos_token = make_input_str(task, vocabs, examples, rev_item2str=rev_item2str, abstract=abstract, tokenizer = tokenizer)
    if verbose: print(text)
    if my_isinstance(tokenizer, LLAMATokenizer):  # add by lxy  avoid  len(text) > max_length
        if len(tokenizer.tokenize(text)) >= max_length:
            return generate(task, nrows - 1, cxt_len, rev_item2str, abstract, plot, verbose, max_length, tokenizer)
    return examples, text, bos_token

def task2str(task):
    vocab_fn, gen_fn, *_ = task
    return f"{fn2str(gen_fn)}[{','.join(str(v) for v in vocab_fn())}]"

def args2str(args):
    strs = []
    for k, v in args.items():
        if type(v) == dict: s = f'{k}=({args2str(v)})' if args2str(v) != '' else ''
        elif v is None: s = ''
        elif type(v) == bool: s = k if v else ''
        elif type(v) == int: s = f'{k}={v}' if v != 0 else ''
        elif type(v) == types.FunctionType: s = f'{k}={v.__name__}'
        else: s = f'{k}={v}'
        strs.append(s)
    return ','.join(s for s in strs if s != '')

def validate_args(task, args, trans_args):
    vocab_fn, gen_fn, *_ = task
    rels = [vocab.relations[0] for vocab in vocab_fn()]
    if trans_args.get('do_swap_qa') and isinstance(gen_fn, partial) and 'query' in gen_fn.keywords: return False
    if not has_local_hop(task) and args.get('rev_item2str'): return False
    if args['cxt_len'] == 1 and (rels[1].name == 'equal' or rels[0].name != 'equal' or 
                                trans_args.get('do_negate') or trans_args.get('do_g2c')):
        return False
    if rels[1].name == 'sibling' and not trans_args.get('do_g2c'): return False
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
