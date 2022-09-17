import sys
from random import choice, choices, shuffle, sample, randint
import string
import itertools
from functools import partial

sys.path.insert(0, '/nas/xd/projects/PyFunctional')
from functional import seq
from functional.pipeline import Sequence
from fn import _
from collections import namedtuple 

from child_utils import uppercase, lowercase, digits, full_vocab


def verbalize(obj):
    if type(obj) == bool: return 'Yes' if obj else 'No'
    return str(obj)
    
def make_query_str(instruction, query):
    if instruction is None and query is None: return ''
    s = '.'
    if instruction is not None: s = s + ' ' + instruction
    if query is not None:
        if type(query) in [int, bool, str]: query = [query]
        if type(query) == dict:
    #         return '. ' + '{' + ','.join([' %s: %s' % (str(k), str(v)) for k, v in query.items()]) + ' }'
            s = s + ' ' + '{' + ','.join([' replace %s with %s' % (str(k), str(v)) for k, v in query.items()]) + ' }'
        elif type(query) in [list,]:
            s = s + ' ' + ' '.join([str(i) for i in query])
    return s

def make_example_str(example, with_instruction=False):
    instruction, l, query, ans = example
    if type(ans) not in [Sequence, list]: ans = [ans]
    ans = [verbalize(a) for a in ans]
    return '%s -> %s' % (' '.join(l) + make_query_str(instruction if with_instruction else None, query), ' '.join(ans))

def sample_rand_len(vocab, k): return sample(vocab, k=randint(1, k))

bop_str = 'Instruction: '
eop_str = '. For example:'

def promptize(s): return bop_str + s + eop_str

def make_input_str(task, nrows=4, ncols=4, full_vocab=None):
    if full_vocab is None: full_vocab = string.ascii_uppercase + string.digits
    transform_fn, vocab_fn, sample_fn, query_fn = task
    instruction = transform_fn.__name__.replace('_', ' ')
    if vocab_fn is None: vocab_fn = lambda: full_vocab
    if query_fn is None: query_fn = lambda *_: None
        
    examples = []
    query = None
    for i in range(nrows):
        vocab = vocab_fn()
        l = sample_fn(vocab, k=ncols)
        query = query_fn(l, vocab, ncols)
        examples.append([instruction, l, query, transform_fn(l, query=query)])

    desc = promptize(instruction) if True else ''
    examples = [make_example_str(e, with_instruction=False) for e in examples]
    text = '\n'.join(examples)
    text = desc + '\n' + text + '\n'
    return text, examples

def ith_element(l, query=None): return seq(l).slice(1, 2)
def ith_group(l, query=None): return seq(l).group_by(_).select(_[1]).slice(1, 2).flatten()#.distinct()# davinci F w/ and wo dist
# def element_at_index(l, query): return seq(l).slice(query, query + 1) # davinci F
def element_at_index(l, query): return seq(l).enumerate().filter(_[0] == query).select(_[1])
def replace(l, query): return seq(l).map(lambda x: query.get(x, x))
def replace_with_the_other(l, query): # davinci F
    query = {k: (set(l) - {k}).pop() for k in l}
    return replace(l, query)
def replace_all_with(l, query): return seq(l).map(lambda x: query)  # davinci F?!
def interleave_with(l, query): return seq(l).flat_map(lambda x: [x, query])  # davinci T!!
def unique_elements(l, query=None): return seq(l).distinct() # davinci F
def how_many_unique_elements(l, query=None): return seq(l).distinct().len()  # davinci F
def how_many(l, query): return seq(l).filter(_ == query).len() # davinci F
def select_same_as(l, query): return seq(l).filter(_ == query) # simpler version of how_many. davinci F
def select_same_number_as(l, query): return seq(l).group_by(_).select(_[1]).filter(lambda x: len(x) == len(query)).flatten() # F
def includes(l, query): return seq(l).union(seq(query)).distinct().len() == seq(l).distinct().len() # davinci F
def is_included_by(l, query): return seq(l).difference(seq(query)).empty() # davinci F

tasks = [
    (ith_element,            None,                               sample,    None),
    (ith_group,              None, lambda vocab, k: seq(sample(vocab, k)).map(lambda x:[x]*randint(1, 3)).flatten().list(),None),
    (element_at_index,       lambda: uppercase,              sample,    lambda l,vocab,k: randint(0, min(2,len(l)-1))),
    (replace,                None,                               sample,    lambda l,vocab,k: {choice(l): choice(vocab)}),
    (replace_with_the_other, lambda: sample(full_vocab, 2),   lambda vocab,k: sample(vocab+choices(vocab, k=k-2),k), None),
    (replace_all_with,       None,                               sample_rand_len, lambda l,vocab,k: choice(vocab)),
    (interleave_with,        None,                               sample_rand_len, lambda l,vocab,k: choice(vocab)),
    (unique_elements,        lambda: sample(uppercase, 3),   choices,   None),
    (how_many_unique_elements,lambda: sample(uppercase, 3),  choices,   None),
    (how_many,               lambda: sample(uppercase, 3),   choices,   lambda l,vocab,k: choice(list(set(l)))),
    (select_same_as,         lambda: sample(uppercase, 3),   choices,   lambda l,vocab,k: choice(list(set(l)))),
    (select_same_number_as,  None, lambda vocab, k: seq(sample(vocab, k)).map(lambda x:[x]*randint(1, 3)).flatten().list(),   
     lambda l,vocab,k: [choice(vocab)]*randint(1, 3)),
    (includes,               lambda: sample(uppercase, 6),   sample,    lambda l,vocab,k: sample(vocab, 3)),
    (is_included_by,         lambda: sample(uppercase, 6),   sample,    lambda l,vocab,k: sample(vocab, 5)),
]

def balance(examples, ans_vocab=[True, False]):
    groups = seq(examples).group_by(_[-1]).map(_[1])  # 按ans分组
    assert groups.len() == len(ans_vocab)  # 保证每种ans都出现
    min_cnt = groups.map(lambda x: len(x)).min()
    examples = groups.map(lambda x: sample(x, min_cnt)).flatten().list() # 每组都采样最小个数后去分组
    return sample(examples, len(examples))  # 重新打乱

def all_a(cxt, query):
    SC, CD = cxt  # SC paris: studeng-course relation, CD pairs: course-department function
    ss, d = query  # ss: 学生子集（可以*不止两个学生*），d: 课程
#     return seq(ss).map(lambda s: seq(SC).filter(_[0] == s).map(_[1]).intersection(CD.filter(_[1] == d).map(_[0])).non_empty()).all()
    return (seq(ss)
            .map(lambda s: seq(SC).filter(_[0] == s).map(_[1])  # 学生s选的所有课程
                 .intersection(
                     seq(CD).filter(_[1] == d).map(_[0])) # d系的课程
                 .non_empty())  # s选了d系的课程
            .all())  # 学生子集ss都选了d系的课程

def all_a_sample(vocab, k):
    S_vocab, C_vocab, D_vocab = vocab  # vocabs of students, courses, departments
    k_S, k_C, k_D, k_SC = k  # default values: k_S = 3, k_C = 3, k_D = 2, k_SC = 5
    S, C, D = sample(S_vocab, k_S), sample(C_vocab, k_C), sample(D_vocab, k_D)
    
    while len(set(CD := choices(D, k=k_C))) < k_D: continue  # ds里每个系的课都要出现
    CD = list(zip(C, CD))  # 得到每门课所属的系
    
    all_SC = list(itertools.product(S, C))  # or seq(S).cartesian(C).list()
    while seq(SC := sample(all_SC, k_SC)).map(_[0]).distinct().len() < k_S: continue  # ss里每个学生都要选课
    return SC, CD

def select_distinct(tuples, col): return seq(tuples).map(_[col]).distinct().list()
    
def all_a_query(cxt,vocab,k):
    SC, CD = cxt
    k_S, k_C, k_D, k_SC = k
    S, D = select_distinct(SC, 0), select_distinct(CD, 1)
    k_ss = randint(2, len(S))
    ss = sample(S, k_ss)
    d = choice(D)
    return ss, d



def p2r(p): p = seq(p); return p.zip(p.inits().zip(p.tails()))#.slice(1, p.len() - 1)
def neighbour(direction, k=1): return lambda x: x[direction][k]
def prev(k=1): return neighbour(0, k)
def next(k=1): return neighbour(1, k)
prevs, nexts = __[0][1:], __[1][1:]
beside = lambda x: (x[0][1], x[1][1])

def ith_element(cxt, query): return seq(cxt).slice(1, 2)[0]
def besides(cxt, query): return seq(cxt).difference(query)[0]
# def besides_query(cxt, vocab): return cxt.a(sample, 2), cxt.list()
def get_poset(e): return tuple([p for p in posets if e in p][0])
def special(cxt, query): return seq(cxt).group_by(get_poset).map(__[1]).find(lambda x: len(x) == 1)[0]
# def special_cxt(vocab, k=3): sample(vocab[0], k - 1) + sample(vocab[1], 1)

def after_query(r, p):
    # e = r.dom().init().a(choice)
    e = choice(r.dom().init().tail().list())
    options = r.image(e).map(beside)[0].a(sample, 2)
    return e, options

def before_query(r, p):
    # e = r.dom().tail().a(choice)
    e = choice(r.dom().init().tail().list())
    options = r.image(e).map(beside)[0].a(sample, 2)
    return e, options

def after(r, q): return r.image(q).map(next())[0]
def before(r, q): return r.image(q).map(prev())[0]
def between(r, q): 
    return r.image(q[0]).map(nexts)[0].intersection(r.image(q[1]).map(prevs)[0]).union(
        r.image(q[0]).map(prevs)[0].intersection(r.image(q[1]).map(nexts)[0]))
    
def monotone_map_cxt(vocab):
    P, p = vocab
    R = p2r(P)
    E1 = R.dom().init().tail().a(choice)
    E2 = R.image(E1).map(beside)[0].a(choice)
    return R, E1, E2

def monotone_map_query(cxt, vocab):
    P, p = vocab
    r = p2r(p)
    e1 = r.dom().init().tail().a(choice)
    options = r.image(e1).map(beside)[0]
    return (r, e1), options

def monotone_map(cxt, query, reverse=False):
    R, E1, E2 = cxt
    r, e1 = query
    return r.image(e1).map(
        seq([prev(), next()]).find(lambda f: (E2 in R.image(E1).map(f)[0]) != reverse)  # reverse = not in. too tricky
    )[0]
    
tasks = [
    (ith_element, None, partial(sample, k=3), None),
    (besides, None, partial(sample, k=3), lambda cxt, vocab: (sample(cxt, 2), cxt)),
    (special, lambda: sample(posets[1:3], 2), lambda vocab: sample(sample(vocab[0], 2) + sample(vocab[1], 1), 2 + 1), None),
    
    (after, lambda: choice(closed_posets[:]), p2r, after_query, lambda r: ''),
    (before, lambda: choice(closed_posets[:]), p2r, before_query, lambda r: ''),
    (between, lambda: choice(posets), p2r, lambda r, p: r.image(r.dom().init().tail().a(choice)).map(beside)[0].a(sample, 2), lambda r: ''),
    (partial(monotone_map, reverse=False), lambda: sample(posets, 2), monotone_map_cxt, monotone_map_query),
    (partial(monotone_map, reverse=True), lambda: sample(closed_posets, 2), monotone_map_cxt, monotone_map_query),
]

if __name__ == "__main__":
    input_strs = [make_input_str(tasks[4], nrows=4, ncols=5) for __ in range(n_total)]
    for s in sample(input_strs, 3): print(s)