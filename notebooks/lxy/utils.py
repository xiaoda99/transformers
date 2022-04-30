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
if __name__ == '__main__':
    get_exmaples_number_English(8)
        