## helper variables/functions to work with wcs data
from typing import List
from collections import namedtuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

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

TermData = namedtuple('TermData', ['term', 'abbrev', 'translation'])
LabCoord = namedtuple('LabCoord', ['L', 'a', 'b'])

NUM_CHIPS = 330
NUM_LANGS = 110
ALL_LANGS = [l for l in range(1, 111)]
ALL_CHIPS = [c for c in range(1, 331)]

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
    filename = "color_distances_hist.pdf"
    title = "Pairwise $\Delta E$ color distance"

    fig, ax = plt.subplots()
    positive = distances > 0
    ax.hist(distances[positive].flatten(), bins = 20)

    ax.set_title(title)
    fig.savefig(filename)



        



























