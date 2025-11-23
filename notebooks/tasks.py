from child_utils import *
from child_utils import _cxt2str, _item2str
from child_frames import *
from const import *

def remove_duplicate_values(dicts):
    value_count = {}
    for d in dicts:
        for values in d.values():
            for value in values:
                value_count[value] = value_count.get(value, 0) + 1
    
    duplicates = set(key for key, count in value_count.items() if count > 1)
    for d in dicts:
        for key, values in d.items():
            d[key] = [value for value in values if value not in duplicates]

def process_samples(json_data):
    processed_data = {}
    for sample in json_data["samples"]:
        subject_e = sample["subject"]
        object_e = sample["object"]
        if subject_e[0] in ['.'] or object_e[0] in ['.']: continue
        if subject_e[-1] in ['.']: subject_e = subject_e[:-1]
        if object_e[-1] in ['.']: object_e = object_e[:-1]  # e.g. D.C. -> D.C
        if object_e not in processed_data:
            processed_data[object_e] = []
        processed_data[object_e].append(subject_e)
        remove_duplicate_values([processed_data])
    return processed_data

def countries_of_landmarks():
    countries_of_landmarks.wh = 'which country'
    countries_of_landmarks.name = 'countries of landmarks'
    countries_of_landmarks.sub_wh = 'the country which'
    landmarks = process_samples(json.load(open('maguangyu/landmark_in_country.json')))
    return landmarks, dict(child='a place in',sibling='a landmark in the same country as')

def color_of_fruits():
    color_of_fruits.name = 'color of fruits'
    color_of_fruits.wh = 'which'
    color_of_fruits.sub_wh = 'the fruit which'
    colorOfFruits = process_samples(json.load(open('maguangyu/color_of_fruits.json')))
    # return colorOfFruits,dict(child='skin color',sibling='')
    return colorOfFruits,dict(child='a kind of fruit with skin color of',sibling='')  # XD

def occupations_Of_Persons():
    occupations_Of_Persons.wh = 'who'
    occupations_Of_Persons.name = 'occupations of persons'
    occupations_Of_Persons.sub_wh = 'the person who'
    occupations_Of_Persons.bos = {'child': ' the'}
    return {
        'the actor': ['Leonardo DiCaprio', 'Meryl Streep', 'Tom Hanks', 'Jennifer Lawrence', 'Denzel Washington', 'Julia Roberts', 'Brad Pitt', 'Natalie Portman', 'Johnny Depp', 'Charlize Theron'],
        'the athlete': ['LeBron James', 'Serena Williams', 'Cristiano Ronaldo', 'Usain Bolt', 'Lionel Messi', 'Simone Biles', 'Michael Phelps', 'Roger Federer', 'Tom Brady', 'Alex Morgan'],
        'the musician': ['Beyoncé', 'Taylor Swift', 'Drake', 'Adele', 'Kanye West', 'Rihanna', 'Ed Sheeran', 'Lady Gaga', 'Justin Bieber', 'Eminem'], # should be singersr
        'the scientist': ['Albert Einstein', 'Marie Curie', 'Stephen Hawking', 'Jane Goodall', 'Neil deGrasse Tyson', 'Ada Lovelace', 'Nikola Tesla', 'Carl Sagan', 'Rosie Franklin', 'Richard Dawkins'],
        'the architect': ['Frank Lloyd Wright', 'Zaha Hadid', 'Le Corbusier', 'Ieoh Ming Pei', 'Rem Koolhaas', 'Norman Foster', 'Antoni Gaudí', 'Louis Sullivan', 'Mies van der Rohe', 'Maya Lin'],
        'the author': ['J.K. Rowling', 'Stephen King', 'Agatha Christie', 'George Orwell', 'Toni Morrison', 'Ernest Hemingway', 'Jane Austen', 'Harper Lee', 'J.R.R. Tolkien', 'Mark Twain'], # not very famous?
        'the entrepreneur': ['Elon Musk', 'Jeff Bezos', 'Bill Gates', 'Mark Zuckerberg', 'Oprah Winfrey', 'Steve Jobs', 'Richard Branson', 'Warren Buffett', 'Larry Page', 'Sergey Brin'],
        'the doctor': ['Dr. Anthony Fauci', 'Dr. Sanjay Gupta', 'Dr. Mehmet Oz', 'Dr. Jane Goodall', 'Dr. Michio Kaku', 'Dr. Ben Carson', 'Dr. Neil deGrasse Tyson', 'Dr. Temple Grandin', 'Dr. Gabor Maté', 'Dr. Sylvia Earle'],
        'the lawyer': ['Barack Obama', 'Hillary Clinton', 'Ruth Bader Ginsburg', 'Johnnie Cochran', 'Thurgood Marshall', 'Gloria Allred', 'Clarence Darrow', 'Sonia Sotomayor', 'Alan Dershowitz', 'F. Lee Bailey'],
        'the artist': ['Pablo Picasso', 'Vincent van Gogh', 'Leonardo da Vinci', 'Frida Kahlo', 'Georgia O\'Keeffe', 'Claude Monet', 'Salvador Dalí', 'Andy Warhol', 'Michelangelo', 'Jackson Pollock']
    },dict(child = '',sibling='')

def countries_of_landmarks():
    countries_of_landmarks.wh = 'which country'
    countries_of_landmarks.name = 'countries of landmarks'
    countries_of_landmarks.sub_wh = 'the country which'
    landmarks = process_samples(json.load(open('maguangyu/landmark_in_country_clean.json')))
    return landmarks,dict(child='a landmark in',sibling='a landmark in the same country as')

def sports_played_by_persons():
    sports_played_by_persons.name = 'sports played by persons'
    sports_played_by_persons.wh = 'who'
    sports_played_by_persons.sub_wh = 'the person who'
    sportPlayerPerson = process_samples(json.load(open('maguangyu/person_plays_pro_sport.json')))
    return sportPlayerPerson,dict(child='a player of',sibling='')


# I = Identity; M = Mophism; A = Aggregation; C = CMP; G = GroupBy; N = Negation, l = local
patterns = ['M', 'A?', 'IA', 'MA',
    'IlI', 'MlI', 'IlM', 'MlM', 'IlMlI',
    'IlA', 'MlA', 'IlC', 'MlC', 'AlI', 'GIlI']
# NE types: boy girl planet letter number month day
# TreeSet(name_genders), TreeSet(name_prons), # resolve iu in speech,  item[0]
# BijectSet(place_landmarks), BijectSet(sport_players) # count, item[1]

# locate-and-copy tasks
# has query: [same, coref, ...] * [self, local prev, local next] * [pos, neg]
# fixed query (positional query): ith
# no query: special, duplicate, set diff
# IlMlI

# trash_tasks = [
#     (lambda: [[TreeSet(types_of_things).child], [EqSet(persons).equal]], g2c(MlM_gen),
#         partial(_cxt2str, sep='. ', item2str=lambda item, vocab: f"The {item[0]} is {item[1]}'s"),
#         # lambda query, vocab: f'So does the {query[0]} belong to {query[1]}',
#         lambda query, vocab: f'{query[0]} {query[1]}'#f"So is the {query[0]} {query[1]}'s",
#     ), 
#     (lambda: [[TreeSet(types_of_things).child], [EqSet(persons).equal]], MlM_gen,
#      # partial(_cxt2str, prefix='There are ', item2str=lambda i, _: f"{i[1]}'s {i[0]}"), lambda q, _: f"So whose {q}", '?' # worse 
#      partial(_cxt2str, sep='. ', item2str=lambda i, _: f"The {i[0]} is {i[1]}'s"), lambda q, _: f'So the {q} belongs', 'Ġto'
#     ),
#     (lambda: [[SymSet(person_adjs).equal], [EqSet(persons).equal]], MlM_gen,
#      partial(_cxt2str, item2str=lambda i, _: f'{i[1]} is {i[0]}'), lambda q, _: f'So who is {q}', '?'
#     ),  # gpt-neox good
#     (lambda: [[EqSet(persons).equal], [PoSet(digits).equal], [EqSet(persons).equal]], g2c(MlMlM_gen),
#         partial(_cxt2str, item2str=lambda item, vocab: f'{item[0]} {item[1]}'),
#         lambda query, vocab: f'{query[0]} and {query[1]} are same?'
#     ),
#     (lambda: [[EqSet(persons).equal], [PoSet(digits).equal], [EqSet(persons).equal]], MlMlM_gen,
#         partial(_cxt2str, item2str=lambda item, vocab: f'{item[0]} is {item[1]}'),
#         lambda query, vocab: f'{query} is the same as', # bare query is better for gpt-neox!?
#     ),
#     (lambda: [[EqSet(persons).equal], [BijectSet(city2resident).proj]], MlM_gen,
#      partial(_cxt2str, sep='. ', item2str=lambda i, _: f'{i[0]} lives in {i[1]}'), lambda q, _: f'{q}', 'Ġis'
#     ), # a little worse than capital -> country
# ]

no_query_tasks = [
    (lambda: [TreeSet(types_of_things).use('equal'), TreeSet(types_of_things).use('equal')], partial(MlM_gen, cxt_sample_fn=enumerate_sample, query=1, has_local_hop=False),
     partial(_cxt2str, prefix='There are ', sep=', ', item2str=lambda i, _: f'{wrap_noun(i)}'), lambda q, _: 'Which is in the middle', "?"
    ),
    (lambda: [TreeSet(types_of_things).use('equal'), TreeSet(types_of_things).use('parent')], partial(MlM_gen, cxt_sample_fn=enumerate_sample, query=1, has_local_hop=False),
     partial(_cxt2str, prefix='There are ', sep=', ', item2str=lambda i, _: f'{wrap_noun(i)}'), lambda q, _: 'Which is in the middle', "?"
    ),
    (lambda: [TreeSet(types_of_things).use('equal'), TreeSet(types_of_things).use('equal')], partial(MlM_gen, cxt_sample_fn=grouped_sample, query=1, has_local_hop=False),
     partial(_cxt2str, prefix='There are ', sep=', ', item2str=lambda i, _: f'{wrap_noun(i)}'), lambda q, _: 'Which is different', "?"
    ),
    (lambda: [TreeSet(types_of_things).use('equal'), EqSet(persons).use('equal')], partial(MlM_gen, cxt_sample_fn=grouped_sample, query=1, has_local_hop=True),
     partial(_cxt2str, sep='. ', item2str=lambda i, _: f'{i[1]} has {wrap_noun(i[0])}'), lambda q, _: 'Who has the different thing', "?"
    ),
    (lambda: [TreeSet(types_of_things).use('child'), TreeSet(types_of_things).use('equal')], partial(MlM_gen, cxt_sample_fn=grouped_sample, query=1, has_local_hop=False),
     partial(_cxt2str, prefix='There are ', sep=', ', item2str=lambda i, _: f'{wrap_noun(i)}'), lambda q, _: 'Which is different', "?"
    ),
    (lambda: [TreeSet(types_of_things).use('child'), EqSet(persons).use('equal')], partial(MlM_gen, cxt_sample_fn=grouped_sample, query=1, has_local_hop=True),
     partial(_cxt2str, sep='. ', item2str=lambda i, _: f'{i[1]} has {wrap_noun(i[0])}'), lambda q, _: 'Who has the different thing', "?"
    ),
]

multi_hop_tasks = [
    (lambda: [EqSet(persons).use('equal'), TreeSet(types_of_things).use('equal')], MlMlM_gen,
     partial(_cxt2str, sep='. ', item2str=lambda i, _: f'{i[0]} has {wrap_noun(i[1])}'), lambda q, _: f'Who has the same thing as {q}', "?"
    ),
]

# for task,       do_swap_qa, rev_item2str in product(
#     tasks[0:1],[False,True],[False,True]):
tasks2 = [
    (lambda: [TreeSet(genders_of_persons).use('equal'), PoSet(temporal_posets).use('prev')], MlM_gen,
     partial(_cxt2str, item2str=lambda i, _: [f'{i[0]} arrived {wrap_noun2(i[1])}', f'{wrap_noun2(i[1]).capitalize()} arrived {i[0]}']), lambda q, _: f'{q} arrived just', ' after'
    ), 
    (lambda: [TreeSet(genders_of_persons).use('equal'), PoSet(temporal_posets).use('next')], MlM_gen,
     partial(_cxt2str, item2str=lambda i, _: [f'{i[0]} arrived {wrap_noun2(i[1])}', f'{wrap_noun2(i[1]).capitalize()} arrived {i[0]}']), lambda q, _: f'{q} arrived just', ' before'
    ),
    # (lambda: [TreeSet(genders_of_persons).use('equal'), SymSet(person_adjs).use('equal')], MlM_gen,
    #  partial(_cxt2str, sep='. ', item2str=lambda i, _: f"{i[0]}'s {i[1]}"), lambda q, _: f'{conj()}{q}{xxx_be()}', "" #" is"
    # ),
    (lambda: [TreeSet(genders_of_persons).use('equal'), SymSet(person_adjs).use('opposite')], MlM_gen,
     partial(_cxt2str, sep='. ', item2str=lambda i, _: f"{i[0]}'s {i[1]}"), lambda q, _: f'{conj()}{q}{xxx_be(False)}', "" #" is"
    #  partial(_cxt2str, sep='. ', item2str=lambda i, _: f"{i[0]}'s {i[1]}"), lambda q, _: f'{conj(False)}{q}{xxx_be()}', "" #" is"
    #  partial(_cxt2str, sep='. ', item2str=lambda i, _: f"{i[0]}'s {i[1]}"), lambda q, _: f"So {q} is", " not"
    ), # t: 16-14, somewhat 14-7 # verbose acc: gpj-j > curie-001 > davinci-001 > gpt-neox!? abstract acc: gpt-neox > gpt-j. all poor (inc. davinci-002!)
]

# for task, replace_rel0, replace_rel1, do_swap_qa, do_negate, do_rm_local_hop, do_rm_query, rev_item2str in product(
#     tasks[:], [0, 1],   [0, 1, 2],   [False,True],[False,True],[False,True],[False,True],[False,True]):
#tasks = [
#    (lambda: [TreeSet(genders_of_persons).use('equal'), TreeSet(types_of_things).use('child')], MlM_gen,
#     partial(_cxt2str, item2str=lambda i, _: [f"{i[0]} has {wrap_noun(i[1])}", f"The {i[1]} is {i[0]}'s"]), lambda q, _: f'{q}', " likes"
#    ), # t: 21-5, 15-8, 19. p: 16-7, 18-5, [3-12, 13-7]. p+: 16-7, 16-0. 13-7:induction head qk, thing->type ov
#    (lambda: [TreeSet(genders_of_persons).use('equal'), TreeSet(countries_of_cities).use('child')], MlM_gen,
#     partial(_cxt2str, item2str=lambda i, _: [f'{i[0]} likes {i[1]}', f'{i[1]} attracts {i[0]}']), lambda q, _: f'{q} wants to go', ' to'
#    ), # t: 19-12 >> 16-10 = 12-7
#    (lambda: [TreeSet(genders_of_persons).use('equal'), TreeSet(capabilities_of_things).use('child')], MlM_gen,
#     partial(_cxt2str, item2str=lambda i, _: [f"{i[0]} has {wrap_noun(i[1])}", f"The {i[1]} is {i[0]}'s"]), lambda q, _: f'{q}', ' can'
#    ),  # t: 13-15, not very strong
#    # (lambda: [EqSet(persons).use('equal'), TreeSet(does2did).use('equal')], MlM_gen,
#    #  partial(_cxt2str, sep='. ', item2str=lambda i, _: f'{i[0]} {i[1]}'), lambda q, _: f'{q}', ' usually'
#    # ),
#    # (lambda: [EqSet(persons).use('equal'), TreeSet(does2did).use('parent')], MlM_gen,
#    #  partial(_cxt2str, sep='. ', item2str=lambda i, _: f'{i[0]} {i[1]}'), lambda q, _: f'{q}', ' usually'
#    # ), # t: copy ov 16-7 + mlp. best performance among all tasks by all models. abstract > verbose
#    # (lambda: [EqSet(persons).use('equal'), TreeSet(en2fr).use('parent')], MlM_gen,
#    #  partial(_cxt2str, sep='. ', item2str=lambda i, _: f'{i[0]} a {i[1]}'), lambda q, _: f'{q}', "'s"
#    # ), # t: good translate ov 16-15, 21-14!
#] # mqy, overwrite tasks variable in jupyter notebook when reload tasks module

neg_tasks = [
    (lambda: [EqSet(persons).use('equal'), TreeSet(types_of_things).use('equal')], MlM_gen,
     partial(_cxt2str, sep='. ', item2str=lambda i, _: f'{i[0]} has {wrap_noun(i[1])}'), lambda q, _: f"No, it is {q}", "'s"
    ),
    (lambda: [EqSet(persons).use('equal'), SymSet(person_adjs).use('equal')], MlM_gen,
     partial(_cxt2str, sep='. ', item2str=lambda i, _: f"{i[0]}'s {i[1]}"), lambda q, _: f'No, {q}', " is"
    ),
]
