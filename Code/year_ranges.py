#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: Sun Jul 21 22:06:52 2019

@michael

#  OPENING DESCRIPTION HERE
"""

# -------------------------
#  START IMPORT STATEMENTS 
# -------------------------
# import math
# -------------------------
#   END  IMPORT STATEMENTS
# -------------------------


# ------------------------
#  START GLOBAL VARIABLES 
# ------------------------
first_year = 1780
last_year  = 2018
increment  = 20
overlap    = increment // 2
# ------------------------
#   END  GLOBAL VARIABLES 
# ------------------------


# ----------------------------
#  START FUNCTION DEFINITIONS 
# ----------------------------

# intended use: include left, exclude right
# half-closed, half-open interval [a, b)
def build_year_ranges(first, last, inc, over):
	year_ranges = []
	for n in range(first, last, over):
		year_ranges.append((n, n + inc))
	return year_ranges
	

# warning: years must have the same index as data
def put_data_under_year_ranges(data, years, year_ranges):

#	assert len(data) == len(years), \
#		"get_content_under_ranges: data and years do not match length"

	# build a dict with keys = year_ranges, with a list for each range
    data_ranges = dict()
    for y in year_ranges:
        data_ranges[y] = []

	# bin all the data by range - each row should fall in two bins, 
	# if ranges are cleanly overlapped
	
	# if data is a list
    for i in range(len(data)):
        for y in year_ranges:
            if y[0] <= years[i] and years[i] < y[1]:
                data_ranges[y].append(data[i])
                # this should happen twice for every entry except 
                # the very oldest and the very newest
	
	# TODO if data is a pandas df

    return data_ranges



# ----------------------------
#   END  FUNCTION DEFINITIONS 
# ----------------------------


# ------------
#  BEGIN MAIN 
# ------------

if __name__ == "__main__":

    import random
    # generate a bunch of fake things 
    
    fake_things = []
    fake_years = []
    some_words = ['this', 'is', 'a', 'word', 'and', 'that', 'ought', 
                  'to', 'be', 'another', 'thing', 'with', 'letters']
                  
    for i in range(500):
        fake_years.append(random.randint(first_year, last_year))
        i = random.randint(0, len(some_words)-1)
        j = random.randint(0, len(some_words)-1)
        fake_things.append(some_words[i] + ' ' + some_words[j])
        
    # and bin them
    bins = build_year_ranges(first_year, last_year, increment, overlap)
    binned_data = put_data_under_year_ranges(fake_things, fake_years, bins)
    
# ------------
#   END  MAIN 
# ------------

