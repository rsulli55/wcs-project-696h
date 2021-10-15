## helper variables/functions to work with wcs data
from typing import List
from collections import namedtuple

import numpy as np
import pandas as pd

termdf = pd.read_csv('term.txt', sep='\t', header=None, names=[
                     'language', 'speaker', 'chip', 'term_abbrev'],
                     na_filter=False)
dictdf = pd.read_csv('dict.txt', sep='\t', skiprows=[0], names=[
                     'language', 'term', 'translation', 'term_abbrev'],
                     na_filter=False)
chipdf = pd.read_csv('chip.txt', sep='\t', names=[
                     'chip', 'letter', 'number', 'letternumber'])

TermData = namedtuple('TermData', ['term', 'abbrev', 'translation'])

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
        if terms.shape[0] > 1:
            print("LANG {} ABBREV {} has more than one term".format(language, abbrev))
        term = terms.min()
        translation = (lang_dict.loc[subset, 'translation']).iloc[0]
        term_data = TermData(term=term, abbrev=abbrev, translation=translation)
        term_map[term] = term_data

    return term_map

def build_word_count(language: int):
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

