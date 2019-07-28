#!/usr/bin/env python
# coding: utf-8

# In[6]:


from __future__ import unicode_literals
import json
import glob

import pickle
import os
import re
import pandas as pd
from pprint import pprint

import glob
# from lxml import html


from bs4 import BeautifulSoup

files = glob.glob('scotus/*json')

            
attributes_list = ['absolute_url', 'author', 
                   'author_str', 'cluster', 
                   'date_created','date_modified', 
                   'download_url', 'extracted_by_ocr',
                   'html','html_columbia', 
                   'html_lawbox','html_with_citations', 
                   'id','joined_by', 'local_path', 
                   'opinions_cited','page_count','per_curiam','plain_text','resource_uri', 'sha1', 'type']


attributes = {'absolute_url': 'unknown','author': 'unknown','author_str': 'unknown','cluster': 'unknown','date_created': 'unknown','date_modified': 'unknown','download_url': 'unknown','extracted_by_ocr': 'unknown','html': 'unknown','html_columbia': 'unknown','html_lawbox': 'unknown','html_with_citations': 'unknown','id': 'unknown','joined_by': 'unknown','local_path': 'unknown','opinions_cited': 'unknown','page_count': 'unknown','per_curiam': 'unknown','plain_text': 'unknown','resource_uri': 'unknown','sha1': 'unknown','type': 'unknown'}


class EmptyDict(dict):
    def __missing__(self, key):
        return ''

cases = []
for file in files:
    with open(file, 'r') as f:
        data = json.loads(f.read(), object_hook=EmptyDict)
        cases.append(data)

df = pd.DataFrame(cases)

## CLEAN OUT HEADERS, APPENDICES AND TABLES

appendix_and_tables = re.search(r"Appendix [A-Z] to(.)+?(Table \d)", df.html_with_citations.iloc[0], re.S).group()

re_appx = re.compile(r"Appendix [A-Z] to ,opinion of.*|APPENDIXES TO OPINION.*", re.S)
#us_const = r"U(U\. S\.) Const\.,? (((a|A)rt\.?|(a|A)mend\.?|(p|P)mbl\.?|(p|P)reamble)( ?[XVI]+))?((, (s|S|&sect;|&#167) ([0-9]+)) ?(, cl\. ([0-9]+)\.?)?)"
amendments = re.compile(r"((first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|twentienth|twenty(-)?first|twenty(-)?second|twenty(-)?third|twenty(-)?fourth|twenty(-)?fifth|twenty(-)?sixth|twenty(-)?seventh))+( amendment)( ?[XVI]+)", re.I)


def compile_regexp():
    f_supp = r"([0-9]+ (F\. Supp\. 2d\.|F\. ?Supp\.) [0-9]+)"
#    us_const = r"(U\. S\.) Const\.,? (((a|A)rt\.?|(a|A)mend\.?|(p|P)mbl\.?|(p|P)reamble)( ?[XVI]+))?((, (s|S|&sect;|&#167) ([0-9]+)) ?(, cl\. ([0-9]+)\.?)?)"
def remove_appendices(string):
    regex = re.compile(r"Appendix [A-Z] to ,opinion of.*|APPENDIXES TO OPINION.*", re.S)
    replacement = ''
    string = re.sub(regex, '', string)
    return string
def remove_syllabus_and_headers(string):
        string = re.sub(r'.+?(?=Opinion of)', '', string, count=1) # removes syllabus
        string = re.sub(r"/\.x", "", string)
        string = re.sub(r'NOTICE.*?to press\.', '', string, count=1) # removes slip op notice
        string = re.sub(r"SUPREME COURT OF THE UNITED STATES[_\s]+No\. \d+–\d+[_\s]+", "", string) # remove header
        string = re.sub(r"(\b\d+)?([\s]*)?((Cite as:)? \d{2,3} U\. S\. (\d|_){4} \(\d{4}\))[\s]+\d+[\s]+(Opinion of [A-Z]+, (C\.)? J\.)?(, )?(concurring|dissenting)?(in part)?(and)?(concurring|dissenting)?(in part)?(in judgment)?[\s]+?", "", string) # partial header
        string = re.sub(r"(\b\d+)?([\s]*)?((Cite as:)? \d{2,3} U\. S\. (\d|_){4} \(\d{4}\))[\s]+\d+[\s]+(Opinion of [A-Z]+, (C\.)? J\.)?[\s]+?[A-Z\.,\s]+(, )?(concurring|dissenting)?(in part)?(and)?(concurring|dissenting)?(in part)?[\s]+?(in judgment)?", "", string)
        string = re.sub(r"\b[\d\s]+([A-Z'\s]+ v\. [A-Z'’\s]+)[\s]+(Syllabus)?[\s]+", "", string) # inline citations
        #string = re.sub(r"\s+Cite as: ?\d+ U\. ?S\. ?[_\d]+ \(\d{4}\)?\s+[\d]+?\s+?", '', string, re.S) # removes citation headers
        #string = re.sub(r"[\d]\s.*?Opinion of.*?, (C\. )?J\.", '', string, re.S) # Removes inline citations
        #string = re.sub(r"(?<=SUPREME COURT OF THE UNITED STATES)(.*?, APPELLANT(S)?) v\. .*?(?=APPEAL FROM)", '', string)
        #string = re.sub(r"SUPREME COURT OF THE UNITED STATES.*?\d{4}\]", '', string, re.S)
        return string
        #[\s]+?[A-Z\.,\s]+

def court_soup(opinion):
    soup = BeautifulSoup(opinion)
    opinion = soup.text
    opinion = re.sub(r'(\n)+', '', opinion)
    opinion = re.sub(r'(\n)', ' ', opinion)
    return opinion


# Remove HTML tags from HTML with citations column

soup = BeautifulSoup(df.iloc[64029].html_with_citations)
soup.text

df['clean_text'] = df['html_with_citations'].apply(lambda x: court_soup(x))


# Drop columns that aren't helping (because they're null, have low/no variance, are not germane to this project, or are available elsewhere)


to_drop = ['date_created','author_str','date_created',
           'date_modified','download_url','extracted_by_ocr','html_columbia',
           'html_lawbox','local_path','opinions_cited','joined_by','resource_uri','sha1','type']

df.drop(to_drop, axis=1, inplace=True)

# Get citation data

cites = glob.glob('data/scotus_clusters/*.json')

citations = []
for cite in cites:
    with open(cite, 'r') as f:
        data = json.loads(f.read(), object_hook=EmptyDict)
        citations.append(data)

citedf = pd.DataFrame(citations)


# get amendments

first = df['clean_text'].str.findall(r"(first amend(\.|ment)?|1st amend(\.|ment)?|(U(\.)? ?S(\.)?)? const(.)? ?amend(\.|ment)? I\b|amend(\.|ment) I\b)", re.I)
    #r"(first amend(\.|ment)|1st amend(\.|ment)|U(\.)? ?S(\.)? const(.)?amend(\.|ment) I)", re.I|re.S)

first = first.apply(lambda x: len(x) > 0)
has_first = df[first].copy()
has_first


# 1839
fifth = df['clean_text'].str.findall(r"(fifth amend(\.|ment)?|5th amend(\.|ment)?|(U(\.)? ?S(\.)?)? const(.)? ?amend(\.|ment)? V\b|amend(\.|ment) V\b)", re.I|re.S)
fifth = fifth.apply(lambda x: len(x) > 0)
has_fifth = df[fifth].copy()
has_fifth



# 855
sixth = df['clean_text'].str.findall(r"(sixth amend(\.|ment)?|6th amend(\.|ment)?|(U(\.)? ?S(\.)?)? const(.)? ?amend(\.|ment)? VI\b|amend(\.|ment) VI\b)", re.I|re.S)
sixth = sixth.apply(lambda x: len(x) > 0)
has_sixth = df[sixth].copy()
has_sixth


# 949
fourth = df['clean_text'].str.findall(r"(fourth amend(\.|ment)?|4th amend(\.|ment)?|(U(\.)? ?S(\.)?)? const(.)? ?amend(\.|ment)? IV\b|amend(\.|ment) IV\b)", re.I|re.S)
fourth = fourth.apply(lambda x: len(x) > 0)
has_fourth = df[fourth].copy()
has_fourth

# 163
second = df['clean_text'].str.findall(r"(\bsecond amend(\.|ment)?|2nd amend(\.|ment)?|(U(\.)? ?S(\.)?)? const(.)? ?amend(\.|ment)? II\b|amend(\.|ment) II\b)", re.I|re.S)
second = second.apply(lambda x: len(x) > 0)
has_second = df[second].copy()
has_second

amendments_df = pd.concat([has_first, has_fifth, has_sixth, has_fourth, has_second])

# 4579
#fourteenth = df['clean_text'].str.findall(r"(fourteenth amend(\.|ment)?|14th amend(\.|ment)?|(U(\.)? ?S(\.)?)? const(.)? ?amend(\.|ment)? XIV\b|amend(\.|ment) XIV\b)", re.I|re.S)
#fourteenth = fourteenth.apply(lambda x: len(x) > 0)
#has_fourteenth = df[fourteenth].copy()
#has_fourteenth

# ## Prepare dataframe for processing

#cluster_keys = has_fourteenth.cluster.str.extractall(r'(?<=\/)(\d+)(?=\/)')
#cluster_keys = cluster_keys.unstack()
#cluster_keys.columns = ['cluster_key']
#has_fourteenth['cluster_key'] = cluster_keys
#has_fourteenth['cluster_key'] = has_fourteenth['cluster_key'].apply(lambda x: int(x))


# Merge citations with cases
cases_df = pd.merge(amendments_df, citedf,
    how='left',
    left_on='id',
    right_on='id')

to_drop_2 = ['absolute_url_x',
'cluster',
'html_with_citations',
'html',
'plain_text',
'absolute_url_y',
'attorneys',
'blocked',
'case_name_full',
'case_name_short',             
'date_blocked',
'date_created',
'date_filed_is_approximate',
'date_modified',
'federal_cite_one',
'federal_cite_three',
'federal_cite_two', 
'neutral_cite',     
'scotus_early_cite',
'slug',
'specialty_cite_one',                                                            
'state_cite_one',                                                                
'state_cite_regional',                                                          
'state_cite_three',                                                              
'state_cite_two',
'page_count',
'procedural_history',
'syllabus',
'westlaw_cite']      


# Drop columns
cases_df.drop(to_drop_2, axis=1, inplace=True)


# cases_df["year"] = cases_df["date_filed"].apply(lambda x: x[:4]).apply(lambda x: int(x))


with open("amendments_df.pickle", "wb") as t:
    pickle.dump(cases_df, t)
t.close()


# In[38]:


amendments_df.to_csv("amendments_df.csv", index=False)


# ## Preprocessing

import string

import nltk
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer # or LancasterStemmer, RegexpStemmer, SnowballStemmer
from nltk.stem.wordnet import wordnet, WordNetLemmatizer

# ### Tokenize, parse, POS tag, lemmatize, NER, stop words


#import lexnlp.nlp.en.transforms.tokens
import lexnlp
import lexnlp.extract.en.courts
from typing import Generator

import regex as re
from reporters_db import EDITIONS, REPORTERS

__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2019, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/master/LICENSE"
__version__ = "0.2.6"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"

CITATION_PTN = r"""
(?:[\s,:\(]|^)
(
(\d+)\s+
({reporters})\s+
(\d+)
(?:,\s+(\d+(?:\-\d+)?))?
(?:\s+\((.+?)?(\d{{4}})\))?
)
(?:\W|$)
""".format(reporters='|'.join([re.escape(i) for i in EDITIONS]))
CITATION_PTN_RE = re.compile(CITATION_PTN, re.IGNORECASE | re.MULTILINE | re.DOTALL | re.VERBOSE)


def get_citations(text, return_source=False, as_dict=False) -> Generator:
    """
    Get citations.
    :param text:
    :param return_source:
    :param as_dict:
    :return: tuple or dict
    (volume, reporter, reporter_full_name, page, page2, court, year[, source text])
    """
    #https://github.com/freelawproject/reporters-db/blob/master/reporters_db/data/reporters.json
    for source_text, volume, reporter, page, page2, court, year            in CITATION_PTN_RE.findall(text):
        try:
            reporter_data = REPORTERS[EDITIONS[reporter]]
            reporter_full_name = ''
            if len(reporter_data) == 1:
                reporter_full_name = reporter_data[0]['name']
            elif year:
                for period_data in reporter_data:
                    if reporter in period_data['editions']:
                        start = period_data['editions'][reporter]['start'].year
                        end = period_data['editions'][reporter]['end']
                        if (end and start <= int(year) <= end.year) or start <= int(year):
                            reporter_full_name = period_data['name']
            item = (int(volume),
                    reporter,
                    reporter_full_name,
                    int(page),
                    page2 or None,
                    court.strip(', ') or None,
                    int(year) if year.isdigit() else None)
            if return_source:
                item += (source_text.strip(),)
            if as_dict:
                keys = ['volume', 'reporter', 'reporter_full_name',
                        'page', 'page2', 'court', 'year', 'citation_str']
                item = {keys[n]: val for n, val in enumerate(item)}
            yield item
        except KeyError:
            pass

def stop_citation_noise(text):
    stops = []
    for cite in get_citations(text, return_source=True, as_dict=True):
        # if len(cite['citation_str']) < 70:
            # text = text.replace(cite["citation_str"], " ")
            #    cite['citation_str'] = cite['citation_str'].replace(str(cite['volume']) + cite['reporter'] + str(cite['page']), " ")
            # stops.append(cite["citation_str"])        
        buildcite = str(cite['volume']) + " " + cite['reporter'] + " " + str(cite['page'])
        if cite['page2']:
            buildcite += ', '+str(cite['page2'])
        stops.append(buildcite)
        
    return stops
        
def replace_citations(text):
    stops = stop_citation_noise(text)
    for stop in stops:
        text = text.replace(stop, " ")
    return text


default_lemmatizer = WordNetLemmatizer()
default_stemmer = PorterStemmer()
default_stopwords = set(stopwords.words('english')) 

def get_wordnet_pos(word):
# Map POS tag to first character lemmatize() accepts
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN) # it's a noun if it's not found

def tokenize_text(text):
    return [w for s in sent_tokenize(text) for w in word_tokenize(s)]

def remove_special_characters(text, characters=string.punctuation.replace('-', '')):
    tokens = tokenize_text(text)
    pattern = re.compile('[{}]'.format(re.escape(characters)))
    return ' '.join(filter(None, [pattern.sub('', t) for t in tokens]))

def lemmatize_text(text, lemmatizer=default_lemmatizer):
    tokens = tokenize_text(text)
    return ' '.join([lemmatizer.lemmatize(t, get_wordnet_pos(t)) for t in tokens])

def stem_text(text, stemmer=default_stemmer):
    tokens = tokenize_text(text)
    return ' '.join([stemmer.stem(t) for t in tokens])

def remove_stopwords(text, stop_words=default_stopwords):
    tokens = [w for w in tokenize_text(text) if w not in stop_words]
    return ' '.join(tokens)
    
    # cleaning pipeline in this function: 
    # remove extra spaces, lowercase, remove stopwords, stem_or_lem
    
def clean_text(text, stem_or_lem = 'lem'):
    text = replace_citations(text)
    #text = text.strip(' ') # strip whitespaces
    text = re.sub(r"[\d]+", " ", text)
    text = text.lower() # lowercase
    text = remove_special_characters(text) # remove punctuation and symbols
    text = remove_stopwords(text) # remove stopwords
    if stem_or_lem == 'stem':
        text = stem_text(text) # stemming
    elif stem_or_lem == 'lem':
        text = lemmatize_text(text) # lemmatizing
    else: # intentionally breaking the argument so neither occurs
        pass 

    return text

amendments_df['corpora'] = amendments_df['clean_text'].apply(lambda x: clean_text(x))


amendments_df.to_csv("cleaned_corpora.csv",index=False)


# amendments_df.sort_values(by="year")


# ## Create overlapping year ranges


#first_year = 1780
#last_year  = 2018
#increment  = 20
#overlap    = increment // 2
# ------------------------

# ------------------------

# intended use: include left, exclude right
# half-closed, half-open interval [a, b)
#def build_year_ranges(first, last, inc, over):
#    year_ranges = []
#    for n in range(first, last, over):
#        year_ranges.append((n, n + inc))
#    return year_ranges


# warning: years must have the same index as data
#def put_data_under_year_ranges(data, years, year_ranges):

    # assert len(data) == len(years), \
    # "get_content_under_ranges: data and years do not match length"

    # build a dict with keys = year_ranges, with a list for each range
#    data_ranges = dict()
#    for y in year_ranges:
#        data_ranges[y] = []

    # bin all the data by range - each row should fall in two bins, 
    # if ranges are cleanly overlapped

    # if data is a list
 #   for i in range(len(data)):
 #       for y in year_ranges:
#            if y[0] <= years[i] and years[i] < y[1]:
#                data_ranges[y].append(data[i])
                # this should happen twice for every entry except 
                # the very oldest and the very newest

#    # pandas df

#    return data_ranges

# ------------

#def run_year_range_build(): # main
    
#    cases = []
#    years = []
#    corpora = []
                  
#    for i in range(500):
#        years.append(first_year, last_year)
#        i = 0, len(corpora)-1
#        j = 0, len(corpora)-1
#        cases.append(corpora[i] + ' ' + corpora[j])
        
    # and bin them
#    bins = build_year_ranges(first_year, last_year, increment, overlap)
#    binned_data = put_data_under_year_ranges(cases, years, bins)


# In[ ]:


with open("amendment_corpora.pickle", "wb") as f:
    pickle.dump(amendments_df, f)
    
amendments_df.to_csv("amendment_corpora.csv")

# --------------------------------------------------------------------

__name__ == "__main__"