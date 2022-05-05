import random
from captum.attr import visualization as viz
from captum.attr import LayerConductance, LayerIntegratedGradients

def get_examples_middle_end(k):
    sets = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    ans = []
    for j in range(k):
        stringss =''
        for i in range(8):
            listss = random.sample(sets,3)
            stringss += ' '.join(listss)+' -> '+listss[1]+'\n'
        ans.append(stringss)
    return ans

def get_exmaples_number_English(k):
    sets = {}
    
    tempstring = 'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty'
    for i,temp in enumerate(tempstring.split(' ')):
        sets[str(i)] = temp
    # print(len(list(sets.keys())))

    ans = []
    for j in range(k):
        stringss =''
        for i in range(8):
            listss = random.sample(list(sets.keys()),1)
            stringss += ' ' + listss[0] + ' -> ' + sets[listss[0]] + '\n'
        ans.append(stringss)
    return ans
def get_exmaples_number_English_reverse(k):
    sets = {}
    
    tempstring = 'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty'
    for i,temp in enumerate(tempstring.split(' ')):
        sets[str(i)] = temp
    # print(len(list(sets.keys())))

    ans = []
    for j in range(k):
        stringss =''
        for i in range(8):
            listss = random.sample(list(sets.keys()),1)
            stringss +=  ' ' + sets[listss[0]] + ' -> ' + listss[0] + '\n'
        ans.append(stringss)
    return ans


def get_exmaples_English_English(k):
    sets = {}
    tempstring = 'one two three four five six seven eight nine ten'
    tempstring2 ='first second third fourth fifth sixth seventh eighth ninth tenth'
    for i,temp in zip(tempstring.split(' '), tempstring2.split(' ')):
        sets[i] = temp
    # print(len(list(sets.keys())))
    ans = []
    for j in range(k):
        stringss =''
        for i in range(8):
            listss = random.sample(list(sets.keys()),1)
            stringss += ' ' + listss[0] + ' -> ' + sets[listss[0]] + '\n'
        ans.append(stringss)
    return ans

def get_exmaples_English_English_reverse(k):
    sets = {}
    tempstring = 'one two three four five six seven eight nine ten'
    tempstring2 ='first second third fourth fifth sixth seventh eighth ninth tenth'
    for i,temp in zip(tempstring.split(' '), tempstring2.split(' ')):
        sets[i] = temp
    # print(len(list(sets.keys())))
    ans = []
    for j in range(k):
        stringss =''
        for i in range(8):
            listss = random.sample(list(sets.keys()),1)
            stringss += ' ' + sets[listss[0]] + ' -> ' + listss[0] + '\n'
        ans.append(stringss)
    return ans 


def get_exmaples_number_first(k):
    sets = {}
    # tempstring = 'one two three four five six seven eight nine ten'
    tempstring2 ='first second third fourth fifth sixth seventh eighth ninth tenth'
    for i,temp in enumerate(tempstring2.split(' ')):
        sets[str(i+1)] = temp
    # print(len(list(sets.keys())))
    ans = []
    for j in range(k):
        stringss =''
        for i in range(8):
            listss = random.sample(list(sets.keys()),1)
            stringss += ' ' + listss[0] + ' -> ' + sets[listss[0]] + '\n'
        ans.append(stringss)
    return ans
def get_exmaples_number_first_reverse(k):
    sets = {}
    # tempstring = 'one two three four five six seven eight nine ten'
    tempstring2 ='first second third fourth fifth sixth seventh eighth ninth tenth'
    for i,temp in enumerate(tempstring2.split(' ')):
        sets[str(i+1)] = temp
    # print(len(list(sets.keys())))
    ans = []
    for j in range(k):
        stringss =''
        for i in range(8):
            listss = random.sample(list(sets.keys()),1)
            stringss += ' ' + sets[listss[0]] + ' -> ' + listss[0] + '\n'
        ans.append(stringss)
    return ans


def get_exmaples_a_A(k):
    sets = {}
    # tempstring = 'one two three four five six seven eight nine ten'
    tempstring ='A B C D E F G H I J K L M N O P Q R S T U V W X Y Z'
    for i,temp in enumerate(tempstring.split(' ')):
        sets[temp.lower()] = temp
    # print(len(list(sets.keys())))
    ans = []
    for j in range(k):
        stringss =''
        for i in range(8):
            listss = random.sample(list(sets.keys()),1)
            stringss += sets[listss[0]] + ' -> ' + listss[0] + '\n'
        ans.append(stringss)
    return ans

def get_exmaples_a_A_reverse(k):
    sets = {}
    # tempstring = 'one two three four five six seven eight nine ten'
    tempstring ='A B C D E F G H I J K L M N O P Q R S T U V W X Y Z'
    for i,temp in enumerate(tempstring.split(' ')):
        sets[temp.lower()] = temp
    # print(len(list(sets.keys())))
    ans = []
    for j in range(k):
        stringss =''
        for i in range(8):
            listss = random.sample(list(sets.keys()),1)
            stringss += listss[0] + ' -> ' + sets[listss[0]] + '\n'
        ans.append(stringss)
    return ans



lxy = [  
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

def get_exmaples_axxx_Axxx(k):
    ans = []
    for j in range(k):
        stringss =''
        listss = random.sample(lxy,8)
        for key,value in listss:
            stringss += key + ' -> ' + value + '\n'
        ans.append(stringss)
    return ans
def get_exmaples_axxx_Axxx_reverse(k):
    ans = []
    for j in range(k):
        stringss =''
        listss = random.sample(lxy,8)
        for key,value in listss:
            stringss += ' '+ value + ' -> ' + key + '\n'
        ans.append(stringss)
    return ans


verb_form =[
    ('sleep','slept'),
    ('go','went'),
    ('talk','talked'),
    ('can','could'),
    ('do','did'),
    ('forget','forgot'),
    ('leave','left'),
    ('are','were'),
    ('begin','began'),
    ('stand','stood'),
    ('take','took'),
    ('have','had'),
    ('fly','flew'),
    ('speak','spoke'),
    ('come','came'),
    ('try','tried'),
    ('want','wanted'),
]
def get_exmaples_verb_form(k):
    ans = []
    for j in range(k):
        stringss =''
        listss = random.sample(verb_form,8)
        for key,value in listss:
            stringss += ' ' + key + ' -> ' + value + '\n'
        ans.append(stringss)
    return ans

def get_exmaples_verb_form_reverse(k):
    ans = []
    for j in range(k):
        stringss =''
        listss = random.sample(verb_form,8)
        for key,value in listss:
            stringss += ' ' + value + ' -> ' + key + '\n'
        ans.append(stringss)
    return ans

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
def get_exmaples_noun2adj(k):
    ans = []
    for j in range(k):
        stringss =''
        listss = random.sample(noun2adj,8)
        for key,value in listss:
            stringss += ' ' + key + ' -> ' + value + '\n'
        ans.append(stringss)
    return ans
def get_exmaples_noun2adj_reverse(k):
    ans = []
    for j in range(k):
        stringss =''
        listss = random.sample(noun2adj,8)
        for key,value in listss:
            stringss += ' ' + value + ' -> ' + key + '\n'
        ans.append(stringss)
    return ans
antonyms = [
    ('big', 'small'),
    ('long', 'short'),
    ('fat', 'thin'),
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
]
def get_exmaples_antonyms(k):
    ans = []
    for j in range(k):
        stringss =''
        listss = random.sample(antonyms,8)
        for key,value in listss:
            stringss += ' '+ key + ' -> ' + value + '\n'
        ans.append(stringss)
    return ans
def get_exmaples_antonyms_reverse(k):
    ans = []
    for j in range(k):
        stringss =''
        listss = random.sample(antonyms,8)
        for key,value in listss:
            stringss += ' '+ value + ' -> ' + key + '\n'
        ans.append(stringss)
    return ans

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
def get_exmaples_adj2very(k):
    ans = []
    for j in range(k):
        stringss =''
        listss = random.sample(adj2very,8)
        for key,value in listss:
            stringss += ' '+ key + ' -> ' + value + '\n'
        ans.append(stringss)
    return ans
capabilities = [ # A x can y.
    ('knife', 'cut'),
    # ('computer', 'calculate'),
    ('phone', 'call'),
    ('TV', 'show'),
    ('car', 'drive'),
    ('printer', 'print'),
    ('pen', 'write'),
    ('saw', 'cut'),
    ('oven', 'bake'),
    ('pot', 'boil'),
    ('gun', 'shoot'),
    # ('pan', 'fry'),
    ('brush', 'paint'),
    ('shovel', 'dig'),
    ('hammer', 'hit'),
    ('lamp', 'light'),
    ('fan', 'blow'),
]
def get_exmaples_capabilities(k):
    ans = []
    for j in range(k):
        stringss =''
        listss = random.sample(capabilities,8)
        for key,value in listss:
            stringss += ' '+ key + ' -> ' + value + '\n'
        ans.append(stringss)
    return ans


en2fr = [
    ('apple', 'pomme'),
    ('cat', 'chat'),
    ('banana', 'banane'),
    ('watermelon', 'pastèque'),
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

def get_exmaples_en2fr(k):
    ans = []
    for j in range(k):
        stringss =''
        listss = random.sample(en2fr,8)
        for key,value in listss:
            stringss += ' '+ key + ' -> ' + value + '\n'
        ans.append(stringss)
    return ans
def get_exmaples_en2fr_reverse(k):
    ans = []
    for j in range(k):
        stringss =''
        listss = random.sample(en2fr,8)
        for key,value in listss:
            stringss += ' '+ value + ' -> ' + key + '\n'
        ans.append(stringss)
    return ans

isA = []
def get_examples_isA(k):
    ans = []
    with open('/nas/xd/projects/transformers/notebooks/lxy/test', 'r') as f:
        for line in f.readlines():
            isA.append(line.strip())
    for j in range(k):
        stringss =''
        listss = random.sample(isA,8)
        for key in listss:
            stringss += ' '+ key + '\n'
        ans.append(stringss)
    return ans


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
    ('Korea', 'Seoul'),
    ('the Philippines', 'Manila'),
    ('Portugal', 'Lisbon'),
    ('Switzerland', 'Bern'),
    ('Thailand', 'Bangkok'),
    ('Turkey', 'Ankara'),
    ('Spain', 'Madrid'),
    ('Greek', 'Athens'),
]

def get_exmaples_country2capital(k):
    ans = []
    for j in range(k):
        stringss =''
        listss = random.sample(country2capital,8)
        for key,value in listss:
            stringss += ' '+ key + ' -> ' + value + '\n'
        ans.append(stringss)
    return ans

def get_exmaples_country2capital_reverse(k):
    ans = []
    for j in range(k):
        stringss =''
        listss = random.sample(country2capital,8)
        for key,value in listss:
            stringss += ' '+ value + ' -> ' + key + '\n'
        ans.append(stringss)
    return ans

if __name__ == '__main__':
    get_exmaples_number_English(8)
        