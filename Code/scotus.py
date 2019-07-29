#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 01:39:52 2019

@author: lorajohns
"""
from __future__ import unicode_literals
import json

import os
import re
import pandas as pd
from pprint import pprint

import glob
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

def parse_object(o):
    for key in o:
        if o[key] is None:
            o[key] = 'unknown'
    return o

def read_json(files): ## the reading
    for file in files:
        with open(file, 'r') as f:
            data = json.loads(f.read(), object_hook=parse_object)
        yield data
         
def json_to_list(files):
    records = read_json(files)
    filerecord= []
    for record in records:
        filerecord.append(record)
    return filerecord

class EmptyDict(dict):
	def __missing__(self, key):
		return ''
    
def create_df():
	cases = []

	for file in files:
	    with open(file, 'rb') as f:
	        data = json.loads(f.read(), object_hook = EmptyDict)
	        cases.append(data)
	        
	df = pd.DataFrame(cases)
	return df

df = create_df() 

def print_stats(df):
    pprint(f'Info: {df.info()}')
    pprint(f'Head: {df.head()}')
    pprint(f'Columns: {df.columns}')
    pprint(f'Types: {df.type.value_counts()}')

appendix_and_tables = re.search(r"Appendix [A-Z] to(.)+?", df.html_with_citations.iloc[0], re.S).group()
  
re_appx = re.compile(r"Appendix [A-Z] to ,opinion of.*|APPENDIXES TO OPINION.*", re.S)
  #us_const = r"U(U\\. S\\.) Const\\.,? (((a|A)rt\\.?|(a|A)mend\\.?|(p|P)mbl\\.?|(p|P)reamble)( ?[XVI]+))?((, (s|S|&sect;|&#167) [0-9]+)) ?(, cl\\. ([0-9]+)\\.?)?)
amendments = re.compile(r"((first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|thirteenth|fourteenth|fifeenth|sixteenth|seventeenth|eighteenth|nineteenth|twentienth|twenty(-)?first|twenty(-)?second|twenty(-)?third|twenty(-)?fourth|twenty(-)fifth|twenty(-)?sixth|twenty(-)?seventh))+( amendment)( ?[XVI]+)", re.I)
full_name = re.compile(r"(?<=SUPREME COURT OF THE UNITED STATES)(.*?, APPELLANT(S)?) v\. .*?(?=APPEAL FROM)")
inline_citations = re.compile(r"[A-Z-'\.’\$\s]* v\. [A-Z-'\.’\$\s]*Opinion of.*?, (C\. )?J\.")
#def compile_regexp():
#    f_supp = r"([0-9]+ (F\\. Supp\\. 2d\\.|F\\. ?Supp\\.) [0-9]+)"
#    date_of_argument = re.search(r"Argued ([A-Z][a-z]+ [0-9]+, [0-9]{4})", df.html_with_citations.iloc[0]).group(0)
#    short_name = re.search(r"((?<!\\opinion\\\d)([\w-]+v(.)?[\w-]+)(?!\\))", df.local_path.iloc[0]).group(0).replace('_', ' ')

  #    us_const = r"(U\\. S\\.) Const\\.,? (((a|A)rt\\.?|(a|A)mend\\.?|(p|P)mbl\\.?|(p|P)reamble)( ?[XVI]+))?((, (s|S|&sect;|&#167) [0-9]+)) ?(, cl\\. ([0-9]+)\\.?)?)\"\n",

def jureeka_filters():
    filters = pd.read_csv('citation_regexp.csv')
    filters.dropna(inplace=True)
    filters['content'] = filters['content'].apply(lambda x: re.sub('/ig,', '', x))
    filters['content'] = filters['content'].apply(lambda x: re.sub('/\/\,', '', x)) 

def remove_appendices(string):
    regex = re.compile(r"Appendix [A-Z] to ,opinion of.*|APPENDIXES TO OPINION.*", re.S)
    re.sub(regex, '', string)
    return string

def remove_syllabus_and_headers(string):
        re.sub(r'.+?(?=Opinion of)', '', string, count=1) # removes syllabus
        re.sub(r'NOTICE:.*?to press\.', '', string, count=1) # removes slip op notice
        re.sub(r'Cite as: \d+ U\. ?S\. ?[_\d]+ \(\d{4}\)?', '', string) # removes citation headers
        

 #def find_amendments(string):
	 #amendments = re.compile(r\"((first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|thirteenth|fourteenth|fifeenth|sixteenth|seventeenth|eighteenth|nineteenth|twentienth|twenty(-)?first|twenty(-)?second|twenty(-)?third|twenty(-)?fourth|twenty(-)fifth|twenty(-)?sixth|twenty(-)?seventh))+( amendment)( ?[XVI]+)\", re.I)
     #matches = df[df.column.str.contains(amendments, x)] # ???

def court_soup(opinion):
	soup = BeautifulSoup(opinion)
	opinion = soup.text
	opinion = re.sub(r'(\\n)+', '', opinion)
	opinion = re.sub(r'(\\n)', ' ', opinion)
	return opinion

to_drop = ['date_created','author_str','date_created','date_modified',
           'download_url','extracted_by_ocr','html_columbia','html_lawbox',
           'local_path','opinions_cited','joined_by','resource_uri','sha1',
           'type']
df.drop(to_drop, axis=1, inplace=True)

df['clean_text'] = df['html_with_citations'].apply(lambda x: court_soup(x))
df['clean_text'] = df['clean_text'].apply(lambda x: re.sub(r'\\n', ' ', x))

def make_cite_df():
    cites = os.listdir('data/scotus_clusters/')
    citations = []
    for cite in cites:
        with open(cite, 'r') as f:
            data = json.loads(f.read(), object_hook=EmptyDict)
            citations.append(data)
    df = pd.DataFrame(cites)
    return df
cites = make_cite_df()


# df['clean_text'] = df['html_with_citations'].apply(lambda x: court_soup(x))
# df['clean_text'] = df['clean_text'].apply(lambda x: re.sub(r'\\n', ' ', x))


# Merge dataframes
cases_df = pd.merge(df, cites[['case_name','date_filed','federal_cite_one',
                               'resource_uri','scdb_id','scdb_decision_direction',
                               'scdb_votes_majority','scdb_votes_minority']],
    how='left',
    left_on='cluster',
    right_on='resource_uri')

# Drop dfs