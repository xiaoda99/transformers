from child_utils import *
from child_utils import _cxt2str, _item2str
from child_frames import *
from const import *

# I = Identity; M = Mophism; A = Aggregation; C = CMP; G = GroupBy; N = Negation, l = local
patterns = ['M', 'A?', 'IA', 'MA',
    'IlI', 'MlI', 'IlM', 'MlM', 'IlMlI',
    'IlA', 'MlA', 'IlC', 'MlC', 'AlI', 'GIlI']
# NE types: boy girl planet letter number month day
# TreeSet(name_genders), TreeSet(name_prons), # resolve iu in speech,  item[0]
# BijectSet(place_landmarks), BijectSet(sport_players) # count, item[1]

trash_tasks = [
    (lambda: [[TreeSet(types_of_things).child], [EqSet(persons).equal]], g2c(MlM_gen),
        partial(_cxt2str, sep='. ', item2str=lambda item, vocab: f"The {item[0]} is {item[1]}'s"),
        # lambda query, vocab: f'So does the {query[0]} belong to {query[1]}',
        lambda query, vocab: f'{query[0]} {query[1]}'#f"So is the {query[0]} {query[1]}'s",
    ), 
    (lambda: [[TreeSet(types_of_things).child], [EqSet(persons).equal]], MlM_gen,
     # partial(_cxt2str, prefix='There are ', item2str=lambda i, _: f"{i[1]}'s {i[0]}"), lambda q, _: f"So whose {q}", '?' # worse 
     partial(_cxt2str, sep='. ', item2str=lambda i, _: f"The {i[0]} is {i[1]}'s"), lambda q, _: f'So the {q} belongs', 'Ġto'
    ),
    (lambda: [[SymSet(person_adjs).equal], [EqSet(persons).equal]], MlM_gen,
     partial(_cxt2str, item2str=lambda i, _: f'{i[1]} is {i[0]}'), lambda q, _: f'So who is {q}', '?'
    ),  # gpt-neox good
    (lambda: [[EqSet(persons).equal], [PoSet(digits).equal], [EqSet(persons).equal]], g2c(IlMlI_gen),
        partial(_cxt2str, item2str=lambda item, vocab: f'{item[0]} {item[1]}'),
        lambda query, vocab: f'{query[0]} and {query[1]} are same?'
    ),
    (lambda: [[EqSet(persons).equal], [PoSet(digits).equal], [EqSet(persons).equal]], IlMlI_gen,
        partial(_cxt2str, item2str=lambda item, vocab: f'{item[0]} is {item[1]}'),
        lambda query, vocab: f'{query} is the same as', # bare query is better for gpt-neox!?
    ),
]

tasks = [
    (lambda: [EqSet(persons).use('equal'), TreeSet(types_of_things).use('equal')], MlM_gen,
     partial(_cxt2str, sep='. ', item2str=lambda i, _: f'{i[0]} has {wrap_noun(i[1])}'), lambda q, _: f'{q}', "'s"
    ),
    (lambda: [EqSet(persons).use('equal'), TreeSet(types_of_things).use('parent')], MlM_gen,
     partial(_cxt2str, sep='. ', item2str=lambda i, _: f"{i[0]} has {wrap_noun(i[1])}"), lambda q, _: f'{q}', "'s"
    ), # t: 21-5, 15-8, 19. p: 16-7, 18-5, [3-12, 13-7]. p+: 16-7, 16-0. 13-7:induction head qk, thing->type ov
    (lambda: [EqSet(persons).use('equal'), SymSet(person_adjs).use('equal')], MlM_gen,
     partial(_cxt2str, sep='. ', item2str=lambda i, _: f"{i[0]}'s {i[1]}"), lambda q, _: f'{conj()}{q}{xxx_be()}', "" #" is"
    ),
    (lambda: [EqSet(persons).use('equal'), SymSet(person_adjs).use('opposite')], MlM_gen,
     partial(_cxt2str, sep='. ', item2str=lambda i, _: f"{i[0]}'s {i[1]}"), lambda q, _: f'{conj()}{q}{xxx_be(False)}', "" #" is"
    #  partial(_cxt2str, sep='. ', item2str=lambda i, _: f"{i[0]}'s {i[1]}"), lambda q, _: f'{conj(False)}{q}{xxx_be()}', "" #" is"
    #  partial(_cxt2str, sep='. ', item2str=lambda i, _: f"{i[0]}'s {i[1]}"), lambda q, _: f"So {q} is", " not"
    ), # t: 16-14, somewhat 14-7 # verbose acc: gpj-j > curie-001 > davinci-001 > gpt-neox!? abstract acc: gpt-neox > gpt-j. all poor (inc. davinci-002!)
    (lambda: [EqSet(persons).use('equal'), TreeSet(country2capital).use('equal')], MlM_gen,
     partial(_cxt2str, sep='. ', item2str=lambda i, _: f'{i[0]} lives in {i[1]}'), lambda q, _: f'{q} is', ' from'
    ),
    (lambda: [EqSet(persons).use('equal'), TreeSet(country2capital).use('parent')], MlM_gen,
     partial(_cxt2str, sep='. ', item2str=lambda i, _: f'{i[0]} lives in {i[1]}'), lambda q, _: f'{q} is', ' from'
    ), # t: 19-12 >> 16-10 = 12-7
    # (lambda: [[EqSet(persons).equal], [BijectSet(city2resident).proj]], MlM_gen,
    #  partial(_cxt2str, sep='. ', item2str=lambda i, _: f'{i[0]} lives in {i[1]}'), lambda q, _: f'{q}', 'Ġis'
    # ), # a little worse than capital -> country
    (lambda: [EqSet(persons).use('equal'), TreeSet(capabilities_of_things).use('equal')], MlM_gen,
     partial(_cxt2str, sep='. ', item2str=lambda i, _: f'{i[0]} has {wrap_noun(i[1])}'), lambda q, _: f'{q}', "'s"
    ),
    (lambda: [EqSet(persons).use('equal'), TreeSet(capabilities_of_things).use('parent')], MlM_gen,
     partial(_cxt2str, sep='. ', item2str=lambda i, _: f'{i[0]} has {wrap_noun(i[1])}'), lambda q, _: f'{q}', ' can'
    ),  # t: 13-15, not very strong
    # (lambda: [EqSet(persons).use('equal'), TreeSet(does2did).use('equal')], MlM_gen,
    #  partial(_cxt2str, sep='. ', item2str=lambda i, _: f'{i[0]} {i[1]}'), lambda q, _: f'{q}', ' usually'
    # ),
    # (lambda: [EqSet(persons).use('equal'), TreeSet(does2did).use('parent')], MlM_gen,
    #  partial(_cxt2str, sep='. ', item2str=lambda i, _: f'{i[0]} {i[1]}'), lambda q, _: f'{q}', ' usually'
    # ), # t: copy ov 16-7 + mlp. best performance among all tasks by all models. abstract > verbose
    # (lambda: [EqSet(persons).use('equal'), TreeSet(en2fr).use('parent')], MlM_gen,
    #  partial(_cxt2str, sep='. ', item2str=lambda i, _: f'{i[0]} a {i[1]}'), lambda q, _: f'{q}', "'s"
    # ), # t: good translate ov 16-15, 21-14!
]

neg_tasks = [
    (lambda: [EqSet(persons).use('equal'), TreeSet(types_of_things).use('equal')], MNlM_gen,
     partial(_cxt2str, sep='. ', item2str=lambda i, _: f'{i[0]} has {wrap_noun(i[1])}'), lambda q, _: f"No, it is {q}", "'s"
    ),
    (lambda: [EqSet(persons).use('equal'), SymSet(person_adjs).use('equal')], MNlM_gen,
     partial(_cxt2str, sep='. ', item2str=lambda i, _: f"{i[0]}'s {i[1]}"), lambda q, _: f'No, {q}', " is"
    ),
]