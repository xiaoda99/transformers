frames = [
    # compiled by XD
    [['early'], ['late']],
    [['anterior'], ['posterior']],
    [['dynamic'], ['static']],
    [['mortal'], ['immortal']],
    [['occupied'], ['vacant']],
    [['intelligent', 'wise', 'smart'], ['stupid']],
    [['arrogant'], ['modest', 'timid']],
    [['benevolent'], ['vicious']],
    [['educated', 'knowledgeable'], ['ignorant']],
    [['honest'], ['dishonest']],
    [['healthy'], ['unhealthy']],
    [['correct'], ['incorrect']],
    [['confident'], ['uncertain']],
    # from BATS
    [['abundant'], ['tight', 'insufficient']],
    [['beautiful'], ['grotesque', 'monstrous']],
    [['big', 'large'], ['small']],
    [['wide'], ['narrow']],
    [['short'], ['tall', 'long']],
    [['heavy'], ['light']],
    [['bright'], ['dark']],
    [['hard'], ['soft']],
    [['hot', 'warm'], ['cold', 'cool']],
    [['slow'], ['fast', 'quick']],
    [['cheap', 'inexpensive'], ['expensive', 'dear']],
    [['clear'], ['unclear', 'undefined']],
    [['near', 'close'], ['far', 'distant']],
    [['colorful'], ['neutral', 'faded']],
    [['common', 'ordinary'], ['uncommon', 'extraordinary']],
    [['competent'], ['incompetent', 'inefficient']],
    [['concerned'], ['casual', 'detached']],
    [['artificial', 'unnatural'], ['natural']],
    [['dangerous'], ['safe', 'harmless']],
    [['decisive'], ['hesitant']],
    [['energetic'], ['inactive']],
    [['familiar'], ['unfamiliar', 'strange']],
    [['happy'], ['unhappy', 'melancholy']],
    [['interesting'], ['boring']],
    [['normal'], ['abnormal', 'irregular']],
    [['rich'], ['poor']],
    [['formal'], ['liberal', 'informal']],
    [['willing', 'voluntary'], ['unwilling', 'involuntary']],
    [['young'], ['old', 'aged']],
    [['easy', 'simple'], ['difficult', 'hard']], #, 'challenging', 'uneasy']],
    # from WNLaMPro
    [['new'], ['old']],
    [['many'], ['few', 'little']],
    [['high'], ['low']],
    [['general'], ['specific']],
    [['international'], ['national']],
    [['local'], ['national']],
    [['popular'], ['unpopular']],
    [['northern'], ['southern']],
    # [['married'], ['unmarried']],
    [['good'], ['bad']],
    [['available'], ['unavailable']],
    [['successful'], ['unsuccessful']],
    [['strong'], ['weak']],
    [['limited'], ['unlimited']],
    [['foreign'], ['domestic']],
    [['related'], ['unrelated']],
    [['legal'], ['illegal']],
    [['likely'], ['unlikely']],
    [['rural'], ['urban']],
    [['potential'], ['actual']],
    [['necessary'], ['unnecessary']],
    [['positive'], ['negative']],
    [['internal'], ['external']],
    [['effective'], ['ineffective']],
    [['affected'], ['unaffected']],
    [['experienced'], ['inexperienced']],
    [['instrumental'], ['vocal']],
    [['civilian'], ['military']],
    [['conventional'], ['unconventional']],
    [['usual'], ['unusual']],
    [['endemic'], ['epidemic']],
    [['vertical'], ['inclined']],
    [['automatic'], ['manual']],
    [['accessible'], ['inaccessible']],
    [['emotional'], ['cerebral']],
    [['theoretical'], ['empirical']],
    [['eligible'], ['ineligible']],
    [['valid'], ['invalid']],
    [['sudden'], ['gradual']],
    [['reasonable'], ['unreasonable']],
    [['compatible'], ['incompatible']],
    [['acceptable'], ['unacceptable']],
    [['comfortable'], ['uncomfortable']],
    [['rational'], ['irrational']],
    [['unconscious'], ['conscious']],
    # [['homosexual'], ['heterosexual']],
    [['induced'], ['spontaneous']],
    [['peripheral'], ['central']],
    [['unconstitutional'], ['constitutional']],
    [['lawful'], ['unlawful']],
    [['undesirable'], ['desirable']],
    [['unpredictable'], ['predictable']],
    [['impractical'], ['practical']],
    # [['unequal'], ['equal']],
    [['immoral'], ['moral']],
    [['unconditional'], ['conditional']],
    [['unfavorable'], ['favorable']],
    [['unrealistic'], ['realistic']],
    [['impartial'], ['partial']],
    [['irresponsible'], ['responsible']],
    [['unprepared'], ['prepared']],
    [['insensitive'], ['sensitive']],
    [['unethical'], ['ethical']],
    [['frivolous'], ['serious']],
    [['pessimistic'], ['optimistic']],
    [['believable'], ['incredible']]
]

_person_adjs_old = [
    # [['fat'], ['thin']],
    # [['hot'], ['cold']],
    # [['big'], ['small']],
    # [['insensitive'], ['sensitive']],
    # [['quiet'], ['loud']],  # noisy
    # [['young'], ['old']],
    # [['conscious'], ['unconscious']],
    # [['asleep'], ['awake']],
    # [['male'], ['female']],
    # [['inside'], ['outside']],
    # [['white'], ['black']],
    [['tall'], ['short']],
    [['careful'], ['careless']],
    [['happy'], ['sad', 'unhappy']],
    [['rich'], ['poor']],
    [['fast'], ['slow']],
    [['beautiful'], ['ugly']],
    [['clean'], ['dirty']],
    # [['gentle'], ['harsh']],
    [['strong'], ['weak']],
    [['good'], ['bad']],
    # [['patient'], ['impatient']],  # confuse with patient vs doctor/healthy
    [['honest'], ['dishonest']],  # fraudulent
    [['brave'], ['cowardly']],
    [['popular'], ['unpopular']],
    [['comfortable'], ['uncomfortable']],
    [['optimistic'], ['pessimistic']],
    [['responsible'], ['irresponsible']],
    [['rational'], ['irrational']],
    [['healthy'], ['sick', 'unhealthy']],
    [['friendly'], ['unfriendly']],  # hostile
    [['interesting'], ['boring', 'uninteresting']],
    [['safe'], ['dangerous']],  # harmless, safe is not good according to 16-14
    [['knowledgeable'], ['ignorant']],
    [['active'], ['passive']],
    # by code-davinci-002
    [['generous'], ['stingy']],
    [['loyal'], ['disloyal']],
    [['reliable'], ['unreliable']],
    [['successful'], ['unsuccessful']],
    [['correct'], ['incorrect']],
    [['right'], ['wrong']],
    [['clean'], ['dirty']],
    [['lucky'], ['unlucky']],
    # [['diligent', 'hardworking'], ['lazy']],
    # [['polite'], ['rude', 'impolite']],  # brutal, harsh
    # [['humble'], ['arrogant']],
    # [['clever', 'smart', 'intelligent'], ['stupid']],

    # [['sane'], ['mad', 'insane']],
    # [['light'], ['heavy', 'dark']],
    # [['serious'], ['funny']],
    # [['messy', 'untidy'], ['tidy', 'neat', 'clean']],
    # [['cerebral'], ['emotional']],
    # [['hesitant'], ['decisive']],
    # [['selfish'], ['selfless']],
    # [['determined'], ['indecisive']],
    # [['insecure'], ['confident']],
    # [['single'], ['married']],
    # [['shy'], ['outgoing']],
    # [['kind'], ['cruel']],
]

_person_adjs = [
    # [['fat'], ['thin']],
    # [['hot'], ['cold']],
    # [['big'], ['small']],
    # [['insensitive'], ['sensitive']],
    # [['quiet'], ['loud']],  # noisy
    # [['young'], ['old']],
    # [['conscious'], ['unconscious']],
    # [['asleep'], ['awake']],
    # [['male'], ['female']],
    # [['inside'], ['outside']],
    # [['white'], ['black']],
    [['careful', 'cautious'], ['careless', 'incautious']],
    [['happy', 'glad'], ['sad', 'unhappy']],
    [['rich', 'wealthy'], ['poor', 'impoverished']],
    [['clean', 'splotless'], ['dirty', 'filthy']],  # messy
    # [['tidy'], ['untidy']],
    [['honest', 'candid'], ['dishonest', 'fraudulent']],
    [['brave', 'bold', 'adventurous', 'daring'], ['cowardly', 'timid']],
    [['healthy', 'fine'], ['sick', 'unhealthy']],  # fit, well
    [['friendly', 'affable'], ['unfriendly', 'hostile']],
    [['interesting', 'fascinating'], ['boring', 'uninteresting']],  # amusing
    # expanded by gpt-4
    [['beautiful', 'attractive', 'pretty'], ['ugly', 'unattractive']],
    [['gentle', 'tender'], ['harsh', 'severe']],
    # [['good', 'virtuous'], ['bad', 'evil']],
    # [['popular'], ['unpopular']],
    [['comfortable', 'cozy'], ['uncomfortable', 'awkward']],
    [['responsible', 'dependable'], ['irresponsible', 'negligent']],
    [['rational', 'logical'], ['irrational', 'unreasonable']],
    [['safe', 'secure'], ['dangerous', 'hazardous']],  # harmless, safe is not good according to 16-14
    [['knowledgeable', 'informed'], ['ignorant', 'uninformed']],
    [['active', 'energetic', 'lively'], ['passive', 'inactive', 'lethargic', 'listless']],
    # given by code-davinci-002 and expaneded by gpt-4
    [['reliable', 'trustworthy'], ['unreliable', 'undependable']],
    [['successful', 'prosperous'], ['unsuccessful', 'failing']],
    [['lucky', 'fortunate'], ['unlucky', 'unfortunate']],
    [['generous', 'benevolent'], ['stingy', 'miserly']],
    [['correct', 'right'], ['incorrect', 'wrong']],
    [['diligent', 'hardworking'], ['lazy', 'indolent']],
    [['courteous', 'polite'], ['rude', 'impolite']],  # brutal, harsh
    [['clever', 'smart', 'intelligent'], ['stupid', 'foolish']],
    # by gpt-4
    [['strong', 'powerful'], ['weak', 'feeble']],
    [['fast', 'quick'], ['slow', 'sluggish']],
    [['tall', 'high'], ['short', 'low']],
    [['full', 'filled'], ['empty', 'vacant']],
    [['quiet', 'silent'], ['noisy', 'loud']],
    [['sharp', 'keen'], ['dull', 'blunt']],
    [['kind', 'compassionate'], ['cruel', 'heartless']],
    [['ambitious', 'driven'], ['apathetic', 'unmotivated']],
    [['curious', 'inquiring'], ['indifferent', 'uninterested']],
    [['loyal', 'faithful'], ['disloyal', 'unfaithful']],
    [['modest', 'humble'], ['arrogant', 'boastful']],
    [['sociable', 'outgoing'], ['introverted', 'reserved']],  # shy
    [['thoughtful', 'considerate'], ['thoughtless', 'inconsiderate']],
    [['patient', 'tolerant'], ['impatient', 'intolerant']],
    [['creative', 'innovative'], ['unimaginative', 'conventional']],
    [['punctual', 'timely'], ['tardy', 'late']],
    [['optimistic', 'positive'], ['pessimistic', 'negative']],
    [['humorous', 'witty', 'funny'], ['serious', 'humorless']],
    [['selfish', 'egotistical'], ['selfless', 'altruistic']],
    [['determined', 'decisive', 'resolute'], ['hesitant', 'indecisive', 'tentative']],
    # [['light', 'bright'], ['dark', 'dim']], # not for person
    # ['warm', 'hot'], ['cool', 'cold']], # not for person

    # [['sane'], ['mad', 'insane']],
    # [['light'], ['heavy', 'dark']],
    # [['serious'], ['funny']],
    # [['cerebral'], ['emotional']],
    # [['insecure'], ['confident']],
    # [['single'], ['married']],
]

def person_adjs(): return _person_adjs
    # ans = []
    # for vec in _person_adjs:
    #     for tmp in vec:
    #         for a in tmp:
    #             ans.append(a.capitalize())
    # return ans

from common_utils import join_lists
def positivities_of_adjs():
    return dict(zip(['positive', 'negtive'], map(join_lists, zip(*_person_adjs))))

adjs = join_lists(positivities_of_adjs().values())