## helper variables/functions to work with wcs data
from typing import List
from collections import namedtuple, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.ticker as ticker
import matplotlib.cm

from scipy.special import softmax

termdf = pd.read_csv('term.txt', sep='\t', header=None, names=[
                     'language', 'speaker', 'chip', 'term_abbrev'],
                     na_filter=False)
dictdf = pd.read_csv('dict.txt', sep='\t', skiprows=[0], names=[
                     'language', 'term', 'translation', 'term_abbrev'],
                     na_filter=False)
chipdf = pd.read_csv('chip.txt', sep='\t', names=[
                     'chip', 'letter', 'number', 'letternumber'])

colordf = pd.read_csv('cnum-vhcm-lab-new.txt', sep='\t', skiprows=1,
                    names=['chip', 'V', 'H', 'C', 'MunH', 'MunV', 'L', 'a', 'b'])

spkrdf = pd.read_csv('spkr-lsas.txt', sep='\t', names=[
                     'language', 'speaker', 'age', 'sex'])

TermData = namedtuple('TermData', ['term', 'abbrev', 'translation'])
LabCoord = namedtuple('LabCoord', ['L', 'a', 'b'])
CdfEntry = namedtuple('CdfEntry', ['term', 'val'])

NUM_CHIPS = 330
NUM_LANGS = 110
ALL_LANGS = [l for l in range(1, 111)]
# 3 languages do not have responses for every chip from every speaker
# found by running find_problem_langs() from below
BAD_LANGS = set([62, 88, 93])
for l in BAD_LANGS:
    ALL_LANGS.remove(l)

ALL_CHIPS = [c for c in range(1, 331)]

CONTESTED_TERM = TermData(term=-2, abbrev='Co', translation='Contested')
NO_TERM = TermData(term=-1, abbrev="No", translation="None")
    
RNG = np.random.default_rng()

# mappings between different indices
chipnum_to_wcsgrid = {}
for c in range(1, 331):
    letter = chipdf.loc[chipdf['chip'] == c].iloc[0]['letter']
    number = chipdf.loc[chipdf['chip'] == c].iloc[0]['number']
    chipnum_to_wcsgrid[c] = (letter, number)

wcsgrid_to_chipnum = {}
for n in chipnum_to_wcsgrid:
    wcsgrid_to_chipnum[chipnum_to_wcsgrid[n]] = n

matrix_to_chipnum = {}
for i, c in enumerate('ABCDEFGHIJ'):
    for j in range(41):
        if (c == 'A' or c == 'J') and j > 0:
            continue
        matrix_to_chipnum[(i,j)] = wcsgrid_to_chipnum[(c, j)]
        
chipnum_to_matrix = {}
for i,j in matrix_to_chipnum:
    chipnum_to_matrix[matrix_to_chipnum[(i,j)]] = (i,j)

matrix_to_wcsgrid = {}
for i,j in matrix_to_chipnum:
    matrix_to_wcsgrid[(i, j)] = chipnum_to_wcsgrid[matrix_to_chipnum[(i, j)]]

def build_term_map(language: int):
    lang_dict = dictdf.loc[dictdf['language'] == language]
    lang_terms = termdf.loc[termdf['language'] == language]
    num_terms = lang_dict['term'].max()
    abbreviations = lang_dict['term_abbrev'].unique()
    term_map = {}

    for abbrev in abbreviations:
        # use the smallest term index for the abbreviation
        subset = lang_dict['term_abbrev'] == abbrev
        terms = lang_dict.loc[subset]['term']
        # if terms.shape[0] > 1:
        #     print("LANG {} ABBREV {} has more than one term".format(language, abbrev))
        term = terms.min()
        translation = (lang_dict.loc[subset, 'translation']).iloc[0]
        term_data = TermData(term=term, abbrev=abbrev, translation=translation)
        term_map[term] = term_data

    return term_map

def build_word_count(language: int):
    """Returns a # of terms by NUM_CHIPS matrix where the (t,c) entry
    counts the number of times term t was used for chip c"""
    lang_dict = dictdf.loc[dictdf['language'] == language]
    lang_terms = termdf.loc[termdf['language'] == language]
    num_terms = lang_dict['term'].max()
    abbreviations = lang_dict['term_abbrev'].unique()
    termabbrev_map = {}
    for abbrev in abbreviations:
        # use the smallest term index for the abbreviation
        subset = lang_dict['term_abbrev'] == abbrev
        termabbrev_map[abbrev] = lang_dict[subset]['term'].min()

    word_count = np.zeros((num_terms, NUM_CHIPS), dtype=int)
    for abbrev in abbreviations:
        for chip in range(NUM_CHIPS):
            subset = (lang_terms['term_abbrev'] == abbrev) & (lang_terms['chip'] == chip + 1)
            word_count[termabbrev_map[abbrev]-1, chip] = lang_terms[subset]['chip'].count()

    return word_count

def build_word_counts(languages: List[int]):
    word_counts = {}
    for lang in languages:
        word_counts[lang] = build_word_count(lang)

    return word_counts

def delta_E_2000(lab1: LabCoord, lab2: LabCoord, k_L: float, k_C: float, k_H: float):
    """Computes the Delta E 2000 difference between the CIELAB coordinates
    `lab1` and `lab2`
    Formula from http://zschuessler.github.io/DeltaE/learn/ with degrees changed to radians"""
    delta_L_prime = lab2.L - lab1.L
    L_bar = (lab1.L + lab2.L) / 2.0 

    C_star = lambda a, b: np.sqrt(a** 2 + b**2)
    C_star_1 = C_star(lab1.a, lab1.b)
    C_star_2 = C_star(lab2.a, lab2.b)
    C_bar = (C_star_1 + C_star_2) / 2.0

    a_prime = lambda a: a + (a / 2.0) * (1 - np.sqrt( (C_bar**7 / (C_bar**7 + 25**7)) ) )
    a_prime_1 = a_prime(lab1.a)
    a_prime_2 = a_prime(lab2.a)

    C_prime = lambda a, b: np.sqrt(a**2 + b**2)
    C_prime_1 = C_prime(lab1.a, lab1.b)
    C_prime_2 = C_prime(lab2.a, lab2.b)
    C_prime_bar = (C_prime_1 + C_prime_2) / 2.0
    delta_C_prime = C_prime_1 - C_prime_2

    # np.arctan2 outputs values in the range [-pi, pi], but we want it in [0, 2pi]
    h_prime = lambda a, b: np.arctan2(b, a) + np.pi
    h_prime_1 = h_prime(a_prime_1, lab1.b)
    h_prime_2 = h_prime(a_prime_2, lab2.b)
    # print(h_prime, h_prime_1, h_prime_2)

    delta_h_prime = 0
    if np.abs(h_prime_1 - h_prime_2) <= np.pi:
        delta_h_prime = h_prime_2 - h_prime_1 
    elif h_prime_2 <= h_prime_1:
        delta_h_prime = h_prime_2 - h_prime_1 + 2*np.pi
    else:
        delta_h_prime = h_prime_2 - h_prime_1 - 2*np.pi

    delta_H_prime = 2 * np.sqrt(C_prime_1 * C_prime_2) * np.sin(delta_h_prime / 2.0)
    H_prime_bar = 0
    if np.abs(h_prime_1 - h_prime_2) > np.pi:
        H_prime_bar = (h_prime_1 + h_prime_2 + 2*np.pi) / 2.0
    else:
        H_prime_bar = (h_prime_1 + h_prime_2) / 2.0

    T = 1 - 0.17*np.cos(H_prime_bar - np.pi / 6.0) + 0.24*np.cos(2*H_prime_bar) + \
            0.32*np.cos(3*H_prime_bar + 6 / 180.0) - 0.2*np.cos(4*H_prime_bar - 63 / 180.0)

    S_L = 1 + (0.015 * (L_bar - 50)**2) / np.sqrt(20 + (L_bar - 50)**2)
    S_C = 1 + 0.045 * C_prime_bar
    S_H = 1 + 0.015 * C_prime_bar * T

    R_T = -2 * np.sqrt(C_prime_bar**7 / (C_prime_bar**7 + 25**7)) * \
                np.sin(np.pi/3.0 * np.exp(-( (H_prime_bar - 275/180.0) / (25/180.0))**2 ))


    delta_e_2000 = np.sqrt( (delta_L_prime / (k_L*S_L))**2 + (delta_C_prime / (k_C*S_C))**2  + \
                    (delta_H_prime / (k_H*S_H))**2 + R_T*(delta_C_prime*delta_H_prime/(k_C*S_C*k_H*S_H)))

    return delta_e_2000

def all_pairwise_color_distances():
    """Computes the pairwise delta_E_2000 distances between chips
    Distance between chip i  and chip j (i < j) is stored in index [i-1, j-1]""" 
    distances = np.zeros((NUM_CHIPS, NUM_CHIPS))
    k_L = 1
    k_H = 1
    k_C = 1

    for c1 in range(1, NUM_CHIPS + 1):
        chip1 = colordf[colordf.chip == c1]
        lab1 = LabCoord(chip1.L.iloc[0], chip1.a.iloc[0], chip1.b.iloc[0])
        for c2 in range(c1 +1, NUM_CHIPS + 1):
            chip2 = colordf[colordf.chip == c2]
            lab2 = LabCoord(chip2.L.iloc[0], chip2.a.iloc[0], chip2.b.iloc[0])
            distances[c1-1, c2-1] = delta_E_2000(lab1, lab2, k_L, k_H, k_C)


    return distances

def make_color_distances_hist(distances):
    "Make a histogram of the pairwise color distances"
    filename = "output/color_distances_hist.pdf"
    title = "Pairwise $\Delta E$ color distance"

    fig, ax = plt.subplots()
    positive = distances > 0
    ax.hist(distances[positive].flatten(), bins = 20)

    ax.set_title(title)
    fig.savefig(filename)

def build_adjacency_dict(distances, threshold):
    """make the adjacency dict using the given threshold"""
    edge_relationship = (distances > 0.001) & (distances < threshold)
    indices = np.where(edge_relationship)
    adjacencies = defaultdict(list)
    for i in range(len(indices[0])):
        c1, c2 = indices[0][i]+1, indices[1][i]+1
        adjacencies[c1].append(c2)
        adjacencies[c2].append(c1)

    return adjacencies


def build_border_distances(distances):
    """returns a dictionary of 'border distances':
        the keys are chip matrix positions (i,j)
        the values are Lists of up to four tuples ((n_i, n_j), d)
        where (n_i, n_j) is the neighbor chip matrix position and d is the distance
        between (i, j) and (n_i, n_j) provided in `distances`"""
    total_rows = len('ABCDEFGHIJ')
    total_cols = 41

    border_distances = {}
    for i, c in enumerate('ABCDEFGHIJ'):
        jvals = range(41)
        if c == 'A' or c == 'J':
            jvals = [0]
        for j in jvals:
            b_dists = []
            chipnum = matrix_to_chipnum[(i, j)]
            
            for (row, col) in [(-1, 0), (0, -1), (0, 1), (1, 0)]:
                nhbr = ((i+row) % total_rows, (j+col) % total_cols)
                if nhbr in matrix_to_chipnum:
                    nhbr_chipnum = matrix_to_chipnum[nhbr]
                    d = max(distances[chipnum-1, nhbr_chipnum-1], distances[nhbr_chipnum-1, chipnum-1])
                    b_dists.append((nhbr, d))

            # special case for row B: A, 0 is above all of B row
            if c == 'B' and j > 0:
                nhbr = (i - 1, 0)
                nhbr_chipnum = matrix_to_chipnum[nhbr]
                d = max(distances[chipnum-1, nhbr_chipnum-1], distances[nhbr_chipnum-1, chipnum-1])
                b_dists.append((nhbr, d))

            # special case for row I: J, 0 is below all of I row
            if c == 'I' and j > 0:
                nhbr = (i + 1, 0)
                nhbr_chipnum = matrix_to_chipnum[nhbr]
                d = max(distances[chipnum-1, nhbr_chipnum-1], distances[nhbr_chipnum-1, chipnum-1])
                b_dists.append((nhbr, d))
                    
            border_distances[(i, j)] = b_dists

    return border_distances

def build_simple_mle(word_count, term_map):
    """Returns 
    1. a map of chipnum -> TermData where each
       chipnum is mapped to the majority term for that chip,
       if there is no majority, it is mapped to CONTESTED_TERM
    2. a list of all the MLE terms found i.e. the BCTs
    3. the number of BCTs"""
    mles = {}
    bcts = set()
    
    for chip in ALL_CHIPS:
        terms = np.argwhere(word_count[:, chip-1] == np.max(word_count[:, chip-1]))
        if len(terms) > 1:
            print("Chip {} was contested".format(chip))
            mles[chip] = CONTESTED_TERM
        else:
            term = terms[0,0]
            mles[chip] = term_map[term+1]
            bcts.add(term_map[term+1])
            
        # print("Row data for chip {}".format(chip))
        # print(word_count[:, chip-1])
        # print("for chip {} got mle = {}".format(chip, mle[chip]))
    
    return mles, list(bcts), len(bcts)

def color_term_grid(term_map, title, filename, lang):
    """term_map should be a dict of chipnums -> TermData
    this function will collapse the term numbers to the range [0, # of unique terms]
    `title` and `filename` should be strings, and `lang` should be an int in the range 1 to 110
    Saves a grid with title Title {title} for Lang {lang} and 
    Filename lang{lang}_filename.png
    """
    # plot the wcs grid with the mle estimate
    fig, ax = plt.subplots()
    num_cols = 41
    num_rows = len('ABCDEFGHIJ')
    X, Y = np.meshgrid(np.arange(num_cols), np.arange(num_rows))
    Z = np.zeros((num_rows, num_cols), dtype=int)
   
    cmap_list = ['xkcd:pale yellow', 'white', 'xkcd:light pink', 'xkcd:peach', 'xkcd:beige',
                 'xkcd:salmon', 'xkcd:lilac', 'xkcd:orangered',
                 'xkcd:tan', 'xkcd:puke green', 'xkcd:rose',
                 'xkcd:seafoam green', 'xkcd:grass green', 'xkcd:baby blue',
                 'xkcd:olive', 'xkcd:forest green', 'xkcd:deep blue',
                 'xkcd:purple blue', 'xkcd:chocolate', 'xkcd:charcoal', 
                 'xkcd:fuschia', 'xkcd:greyish purple']
    if len(np.unique(term_map.values())) > len(cmap_list) - 1:
        print("ERROR: too many cats, cmap does not have enough colors")
    
    # get a set of all mle_terms we will care about
    mle_terms = set([(data.term, data.abbrev) for data in term_map.values()])
    mle_terms.add((NO_TERM.term, NO_TERM.abbrev))
    # sort them and store abbreviation labels
    mle_terms = sorted(mle_terms)
    abbrev_labels = [pair[1] for pair in mle_terms]

    # collapse the terms numbers to the range [0, num_mle_terms]
    sorted_terms = [pair[0] for pair in mle_terms]
    collapsed_terms = {}
    for collapsed, term in enumerate(sorted_terms):
        collapsed_terms[term] = collapsed
    collapsed_counts = defaultdict(int)
    
    for row in range(num_rows):
        for col in range(num_cols):
            if (row, col) in matrix_to_chipnum:
                term = term_map[matrix_to_chipnum[(row, col)]].term
                Z[row, col] = collapsed_terms[term]
                collapsed_counts[collapsed_terms[term]] += 1
            else:
                Z[row, col] = collapsed_terms[NO_TERM.term]
                collapsed_counts[collapsed_terms[NO_TERM.term]] += 1
                # print("using no_mle_val for col {} and row {}".format(col, row))

    
    cmap = ListedColormap(cmap_list[:len(abbrev_labels)])
    mesh = ax.pcolormesh(X, Y, Z, shading='auto', edgecolors='black', cmap=cmap)
    # pcolormesh needs to have data flipped
    ax.invert_yaxis()
    
    # colorbar and ticks alignment
    cbar = fig.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap), ax=ax)
    tick_spacing = 1.0 / len(abbrev_labels)
    ticks = ticker.FixedLocator(np.arange(tick_spacing/2.0, 1.0, tick_spacing))
    cbar.set_ticks(ticks)
    tick_texts = cbar.ax.set_yticklabels(abbrev_labels)
    # tick_texts[0].set_verticalalignment('top')
    # tick_texts[-1].set_verticalalignment('bottom')
    cbar.ax.tick_params(length=0) # remove the tick marks

    # setup grid labels and title
    ax.set_title(f"{title} for Language {lang}")
    ax.set_xticks(np.arange(num_cols))
    ax.set_yticks(np.arange(num_rows))
    ax.set_xticklabels([str(i) for i in range(41)])
    ax.set_yticklabels([c for c in 'ABCDEFGHIJ'])
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    fig.set_size_inches((36, 8))
    # plt.show()
    plt.savefig(f"output/lang{lang}_{filename}.png")


def build_maj_vote_model(word_count):
    """ builds the parameters for the majority vote model:
        returns a dict which maps chipnum -> (dict of term_num -> weight)
        if there is a simple majority for chipnum c then the value at c will
        be a one element dict with key being the simple majority term and weight being 1
        if there is not a simple majority for chipnum c then the value at c will represent
        a uniform distribution over all terms which were spoken the maximal amount of times"""
        
    model = defaultdict(dict)
    for chip in ALL_CHIPS:
        terms = np.argwhere(word_count[:, chip-1] == np.max(word_count[:, chip-1]))
        for t in terms:
            term_num = t[0] + 1
            model[chip][term_num]=1.0/len(terms)
        if len(terms) > 1:
            print(f"For chip={chip} model[chip] is {model[chip]}")

    return model

def build_simple_model(word_count):
    """ builds the parameters for the majority vote model:
        returns a dict which maps chipnum -> (dict of term_nums -> weight)
        the weight is calculated based on the frequency of times term_num was used for chipnum"""
    
    num_terms = word_count.shape[0]
    orig_num_responses = np.sum(word_count[:, 0])
    model = defaultdict(dict)
    for chip in ALL_CHIPS:
        terms = np.argwhere(word_count[:, chip-1] > 0)
        num_responses = np.sum(word_count[terms, chip-1])
        # if orig_num_responses != num_responses:
        #     print(f"Orig_num_responses {orig_num_responses} does not match num_responses {num_responses} for chip {chip} and terms {word_count[terms, chip-1]}")
        for t in terms:
            term_num = t[0] + 1
            model[chip][term_num] = word_count[term_num-1, chip-1] / num_responses

    return model

def build_held_out_word_count(language, speakers):
    """ language: is a language number [1, 110]
        speakers: list of speaker numbers to hold out
        returns a matrix, dict pair where
        the matrix is a # of terms by NUM_CHIPS matrix W where 
        W[t-1, c-1] = n iff term t was used n times for chip c 
        the dict has represents the held out speaker data
        so it has keys coming from `speakers` and values are 
        length NUM_CHIPS arrays A where
        A[c-1] = t-1 iff term t was used on chip c
    """
    lang_dict = dictdf.loc[dictdf['language'] == language]
    lang_terms = termdf.loc[termdf['language'] == language]
    num_terms = lang_dict['term'].max()
    abbreviations = lang_dict['term_abbrev'].unique()
    termabbrev_map = {}
    for abbrev in abbreviations:
        # use the smallest term index for the abbreviation
        subset = lang_dict['term_abbrev'] == abbrev
        termabbrev_map[abbrev] = lang_dict[subset]['term'].min()

    word_count = np.zeros((num_terms, NUM_CHIPS), dtype=int)
    for abbrev in abbreviations:
        for chip in range(NUM_CHIPS):
            subset = (lang_terms['term_abbrev'] == abbrev) & (lang_terms['chip'] == chip + 1)
            # remove held out speakers
            subset = subset & ~lang_terms['speaker'].isin(speakers)
            word_count[termabbrev_map[abbrev]-1, chip] = lang_terms[subset]['chip'].count()

    held_out = {}
    for speaker in speakers:
        responses = np.zeros(NUM_CHIPS, dtype=int)
        for chip in range(NUM_CHIPS):
            subset = (lang_terms['speaker'] == speaker) & (lang_terms['chip'] == chip + 1)
            # ignore responses where the term_abbrev was "*"
            if lang_terms[subset]['term_abbrev'].isin(['*']).any():
                # print(lang_terms[subset]['term_abbrev'])
                continue
            if len(lang_terms[subset]['term_abbrev']) != 1:
                print(f"\tIssue building hold out set for language {language}")
                print(f"\tlang_terms[subset]['term_abbrev']) is not one element") 
                print(f"\tWe are looking at speaker {speaker}, and chip {chip}")
                print(f"\t{lang_terms[subset]['term_abbrev']}")
                assert(len(lang_terms[subset]['term_abbrev']) == 1)
            abbrev = lang_terms[subset]['term_abbrev'].iloc[0]
            responses[chip] = termabbrev_map[abbrev]-1
        held_out[speaker] = responses


    return word_count, held_out


def build_rand_set_of_speakers(language, fraction):
    """returns a List of speakers chosen randomly
    without replacement from the set of speakers for language `language`
    the size of the List is determinde by `fraction`"""

    lang_speakers = spkrdf.loc[spkrdf['language'] == language]['speaker']
    rand_set = lang_speakers.sample(frac=fraction)
    return list(rand_set)


def compute_simple_NLL(held_out, model):
    """computes the NLL using model for the chip responses in the dict held_out
    returns NLL, and # of problems where a problem occurs if a chip response has zero probability
    in the model"""
    AVE_NUM_SPKRS = 24
    nll = 0
    total_probs = 0
    for spkr in held_out:
        responses = held_out[spkr]
        for chip in ALL_CHIPS:
            term = responses[chip-1] + 1
            log_prob = 0
            if term not in model[chip]:
                # print(f"Problem with speaker {spkr} for chip {chip}, where response was term {term}")
                total_probs += 1
                log_prob = np.log(1.0/AVE_NUM_SPKRS)
            else:
                log_prob = np.log(model[chip][term])
            nll -= log_prob

    return nll, total_probs

def simple_NLL_experiment(language, num_trials, fraction):
    """performs num_trials experiments where each experiment 
    consists of holding out `fraction` percent of speakers,
    creating the "simple model" for the empirical chip distribution,
    computing the NLL of the held out speakers

    returns the average NLL over all trials, and the average number of problems
    per experiment when computing NLL"""

    total_nll = 0
    total_probs = 0
    for trial in range(num_trials):
        # print(f"Running trial {trial} of experiment for language {language}")
        # print("------------------------------------------------------------------\n")
        ho_speakers = build_rand_set_of_speakers(language, fraction)
        wc, ho = build_held_out_word_count(language, ho_speakers)
        model = build_simple_model(wc)
        nll, probs = compute_simple_NLL(ho, model)
        total_nll += nll
        total_probs += probs

    return total_nll / num_trials, total_probs / num_trials

def all_langs_simple_NLL_experiment(num_trials, fraction):
    """calls simple_NLL_experiment(lang, num_trials, fraction) for each lang
    returns a two numpy arrays, each of lange NUM_LANGS which store
    average nll and average number of problems at respectively at index
    lang -1
    Note: the experiment is not run for the three languages in BAD_LANGS"""
    ave_nlls = np.zeros(NUM_LANGS)
    ave_probs = np.zeros(NUM_LANGS)

    for lang in ALL_LANGS:
        print(f"Running experiment for language {lang}\n\n")
        nll, probs = simple_NLL_experiment(lang, num_trials, fraction)
        ave_nlls[lang-1] = nll
        ave_probs[lang-1] = probs

    return ave_nlls, ave_probs

def find_problem_langs():
    problems = set()
    for language in range(NUM_LANGS):
        print(f"Checking language {language}")
        lang_speakers = spkrdf.loc[spkrdf['language'] == language]['speaker']
        lang_terms = termdf.loc[termdf['language'] == language]
        for speaker in lang_speakers:
            for chip in range(NUM_CHIPS):
                subset = (lang_terms['speaker'] == speaker) & (lang_terms['chip'] == chip + 1)
                if len(lang_terms[subset]['term_abbrev']) != 1:
                    print(f"\tIssue with language {language}, speaker {speaker}, and chip {chip}")
                    print(f"\t{lang_terms[subset]['term_abbrev']}")
                    problems.add(language)

    return problems

def check_langs_data(language):
    print(f"Checking language {language}")
    lang_speakers = spkrdf.loc[spkrdf['language'] == language]['speaker']
    lang_terms = termdf.loc[termdf['language'] == language]
    for speaker in lang_speakers:
        for chip in range(NUM_CHIPS):
            subset = (lang_terms['speaker'] == speaker) & (lang_terms['chip'] == chip + 1)
            if len(lang_terms[subset]['term_abbrev']) != 1:
                print(f"\tIssue with language {language}, speaker {speaker}, and chip {chip}")
                print(f"\t{lang_terms[subset]['term_abbrev']}")




def sample_initial_grid_state(word_count, adjacency_dict, num_samples, response_sample_fraction):
    """ word_count is the word matrix for the language,
        adjacency_dict is the adjacency_dict created by build_adjacency_dict()
        num_samples is how many total samples (of the whole grid) to perform before returning the grid
        response_sample_fraction: for an individual chip sample, how often should we sample from the response set
        the grid has the following interpretation
        grid[c-1] = t-1 iff chip c is labeled term t"""

    grid = np.zeros(NUM_CHIPS, dtype=np.int16)
    response_probs = build_simple_model(word_count)
    response_cdf = {}

    # build initial grid configuration and response cdfs for each chip
    for chip in ALL_CHIPS:
        cdf = []
        val = 0
        probs = response_probs[chip]
        dec_probs = sorted(probs, key=lambda t: probs[t], reverse=True)
        for term in dec_probs:
            val += probs[term]
            cdf.append(CdfEntry(term=term, val=val))

        # cdf[0] stores the most likely term
        grid[chip-1] = cdf[0].term - 1
        response_cdf[chip] = cdf

    for sample in range(num_samples):
        print(f"Working on sample {sample+1} of grid")
        for chip in ALL_CHIPS:
            p = RNG.uniform()
            # sample from response cdf
            if p < response_sample_fraction:
                p = RNG.uniform()
                cdf = response_cdf[chip]
                found = False
                for c in cdf:
                    if p < c.val:
                        grid[chip-1] = c.term - 1
                        found = True
                assert(found)
            
            # otherwise sample proportionally to current labels on chip and neighbors
            else:
                # count the frequency of terms
                terms = defaultdict(int)
                # count ourself
                terms[grid[chip-1]+1] += 1
                num_terms = 1
                neighbors = adjacency_dict[chip]
                for nhbr in neighbors:
                    terms[grid[nhbr-1]+1] += 1
                    num_terms += 1

                # build the cdf over the terms
                cdf = []
                val = 0
                dec_freq = sorted(terms, key=lambda t: terms[t], reverse=True)
                for t in dec_freq:
                    val += terms[t] / num_terms
                    cdf.append(CdfEntry(term=t, val=val))

                # if len(cdf) > 1:
                #     print(f"Chip {chip} has interesting cdf {cdf}")
                p = RNG.uniform()
                found = False
                for c in cdf:
                    if p < c.val:
                        grid[chip-1] = c.term - 1
                        # if len(cdf) > 1:
                        #     print(f"Previously, grid has {grid[chip-1]+1}, now we use {c}")
                        found = True
                assert(found)


    return grid

def mrf_sample_grid(word_count, adjacency_dict, neighbor_weight, grid):
    """Does a single MRF grid sample using `word_count`, `adjacency_dict` and
    `neighbor_weight` from the grid configuration `grid`
    """
    for chip in ALL_CHIPS:
        terms_from_chip = np.argwhere(word_count[:, chip-1] > 0).flatten()
        terms_from_neighbor = np.zeros(len(adjacency_dict[chip]), dtype=int)
        
        for i, nhbr in enumerate(adjacency_dict[chip]):
            terms_from_neighbor[i] = grid[nhbr-1]

        terms_combined = np.union1d(terms_from_chip, terms_from_neighbor)
        term_freqs = np.zeros(terms_combined.shape)

        for i, term in enumerate(terms_combined):
            # count terms from our chip
            term_freqs[i] += word_count[term, chip-1]
            # print(f"For term {term} our chip inc freq by {word_count[term, chip-1]}")

            # count weighted terms from neighbors
            term_freqs[i] += neighbor_weight * np.sum(terms_from_neighbor == term)
            # neighbor_freq = neighbor_weight * np.sum(terms_from_neighbor == term)
            # print(f"For term {term} neighbors inc freq by {neighbor_freq}")

        term_pmf =  softmax(term_freqs)
        new_term = RNG.choice(terms_combined, 1, p=term_pmf)[0]
        # if new_term != grid[chip-1]:
        #     print(f"Sampled new term = {new_term}, for chip {chip}, old term was {grid[chip-1]}")
        
        # update grid
        grid[chip-1] = new_term

    return grid


def mrf_sampler(word_count, adjacency_dict, neighbor_weight, num_restarts,
        burn_in_iterations, num_to_generate):
    """Completes the following sampling process
    For `num_restart` runs:
        1. Samples an initial grid configuration using `sample_initial_grid_state()`
        with arguments `num_samples` = 15, `response_sample_fraction = 0.5`
        2. Using this initial grid configuration, does MRF sampling of the grid for 
        `burn_in_iterations` samples
        3. After the burn in, samples `num_to_generate` grid samples and stores them
    Returns the num_restarts * num_to_generate number of samples collection
    """

    INITIAL_GRID_NUM_SAMPLES = 15
    INITIAL_GRID_RESPONSE_FRAC = 0.5

    samples = []
    
    for restart in range(num_restarts):
        print(f"\nOn restart {restart+1}, sampling initial grid")
        grid = sample_initial_grid_state(word_count, adjacency_dict, 
                INITIAL_GRID_NUM_SAMPLES, INITIAL_GRID_RESPONSE_FRAC)

        for iteration in range(burn_in_iterations):
            print(f"\tOn burn-in iteration {iteration+1}")
            grid = mrf_sample_grid(word_count, adjacency_dict, neighbor_weight, grid)

        print()
        for sample in range(num_to_generate):
            print(f"\tGenerating sample {sample+1}")
            grid = mrf_sample_grid(word_count, adjacency_dict, neighbor_weight, grid)
            samples.append(grid)


    return samples

def build_mrf_model_from_samples(samples):
    """ builds the empirical chip distribution from the grids in samples
        returns a dict which maps chipnum -> (dict of term_nums -> weight)
        the weight is calculated based on the frequency of times term_num was used for chipnum"""
   
    num_samples = len(samples)
    model = defaultdict(dict)
    for chip in ALL_CHIPS:
        terms_for_chip = np.zeros(num_samples, dtype=int)
        for i, sample in enumerate(samples):
            terms_for_chip[i] = sample[chip-1]+1
        
        terms, counts = np.unique(terms_for_chip, return_counts=True)
        print(f"terms = {terms} and counts = {counts}")

        for i, t in enumerate(terms):
            model[chip][t] = counts[i] / num_samples

    return model


def compute_KL_divs(p_model, q_model):
    """Assumes that `p_model` and `q_model` are dicts of the form chipnum -> (termnum -> weight)
    like those obtained from build_mrf_model_from_samples and build_simple_model
    Returns an array of the KL divergences between each model on every chip

    For each chip, we compute KL(p_model[chip] || q_model[chip])"""

    divergences = np.zeros(NUM_CHIPS)

    epsilon = 1E-8

    for chip in ALL_CHIPS:
        p = p_model[chip]
        q = q_model[chip]

        terms = set(p.keys()).union(q.keys())
        print(f"For chip {chip}, terms from both dists are {terms}")

        for term in terms:
            # if p[term] = 0, divergence computation is 0
            if term not in p:
                continue
            # add in small positive if q[term] = 0
            if term not in q:
                print(f"Term {term} was in in distribution Q!")
                q[term] = epsilon
            else:
                divergences[chip-1] += p[term] * np.log(p[term] / q[term])

    return divergences









