#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Requires Python 3.6 or higher, plus the following packages:
For Reynir: tokenizer, reynir
For the ABL deep neural marker: numpy, nltk, gspread, 
    colored, dyNET, google_api_python_client, plotly, tabulate
For Levenshtein distance: edlib
For Kvistur: DAWG-Python
"""


##### Import declarations #####

import os
import re
import math
import subprocess
import edlib
import nltk
nltk.download('punkt')
import argparse
from reynir import Reynir
from tokenizer import tokenize, TOK
import kvistur
import re

#Internal/temp: Code for speed-testing
import sys
from datetime import datetime
#Internal code ends


##### Global constant declarations #####

DEFAULT_PATTERN_FILE = 'default_patterns.txt'
DEFAULT_C_VALUE = 3.0
DEFAULT_L_DISTANCE = 15
DEFAULT_S_RATIO = 1.5

#Internal/temp: Constants used only for temporary test functions
FILE_GOLDSTANDARD_TERMS = 'id-gull-greynir.txt'
FILE_GOLDSTANDARD_LEMMATIZED = 'id-gull-nafnmyndir.txt'
FILE_OUT_RESULTS = 'nidurstodur.txt'


##### Global variable declarations #####

known_term_list = []
known_term_list_roots = []
term_candidate_list = []
pattern_list = []
stop_list = []

#Internal/temp: Variable used only for temporary test code.
gold_standard_list = []


##### Function declarations #####

def load_known_terms(r, file_lemmas, file_all):
    global known_term_list
    
    if( (file_lemmas) and (not file_all) ):
        with open(file_lemmas, "r", encoding="utf-8") as f_l:
            for newline_l in iter(f_l.readline, ''):
                line_l_full = str(newline_l)
                line_l_string = line_l_full.rstrip()
                if( line_l_string ):
                    known_term_list.append(line_l_string)
        f_l.close()
    elif( (not file_lemmas) and (file_all) ):
        line_counter = 0
        with open(file_all, "r", encoding="utf-8") as f_a:
            for newline_a in iter(f_a.readline, ''):
                line_counter += 1
                if( linecounter % 3 == 2 ):
                    line_a_full = str(newline_a)
                    line_a_string = line_a_full.rstrip()
                    if( line_a_string ):
                        known_term_list.append(line_a_string)
        f_a.close()
    else:
        known_term_list = []


def populate_pattern_list(r, pattern_file):
    pattern_l = []

    if( pattern_file ):
        with open(pattern_file, "r", encoding="utf-8") as file_p:
            for line_p_str in file_p:
                line_p_list = [ifd_tag[0] for ifd_tag in line_p_str.split()]
                pattern_l.append(line_p_list)
        file_p.close()

    return pattern_l


def load_roots_from_known_terms():
    global known_term_list
    root_list = []
    
    if( len(known_term_list)>0 ):
        resources = {
            "modifiers": os.path.join(os.path.dirname(__file__), 'resources', 'modifiers.dawg'),
            "heads": os.path.join(os.path.dirname(__file__), 'resources', 'heads.dawg'),
            "templates": os.path.join(os.path.dirname(__file__), 'resources', 'templates.dawg'),
            "splits": os.path.join(os.path.dirname(__file__), 'resources', 'splits.dawg')
        }
        kv = kvistur.Kvistur(**resources)

        for line in known_term_list:
            root_line = []
            line_list = line.split()
            for word in line_list:
                score, tree = kv.decompound(word)
                compound_list = []
                compound_list = tree.get_atoms()
                if( len(compound_list) > 1):
                    root_line.append(compound_list)
            if( len(root_line) > 0):
                root_list.append(root_line)
    
    return root_list


def load_stop_list(stoplist_file):
    stop_l = []

    if( stoplist_file ):
        with open(stoplist_file, "r", encoding="utf-8") as file_s:
            for newline_s in iter(file_s.readline, ''):
                line_s_full = str(newline_s)
                line_s_string = line_s_full.rstrip()
                if( line_s_string ):
                    """
                    line_s_list = [word for word in line_s_string.split()]
                    stop_l.append(line_s_list)
                    """
                    stop_l.append(line_s_string)
                    
        file_s.close()

    return stop_l


def line_tokenize(newline):
    list_of_tokenized_words = []
    
    for token in tokenize(newline):
        kind, txt, val = token
        if kind == TOK.WORD:
            list_of_tokenized_words.append(txt)
    return list_of_tokenized_words


def line_tag(tokenized_list, r):
    str_tokens = " ".join(tokenized_list)
    parsed_tokens = r.parse_single(str_tokens)
    return parsed_tokens


def line_lemmatize(pos_tagged_sentence):
    lemmatized_words = []
    
    if pos_tagged_sentence.tree is not None:
        number_of_tags = len(pos_tagged_sentence.ifd_tags)
        number_of_words = len(pos_tagged_sentence.lemmas)
        lemma_list_with_dashes = pos_tagged_sentence.lemmas
        ifd_tag_list_with_dashes = pos_tagged_sentence.ifd_tags
        
        """
        Reynir sometimes inserts dashes ("-") into lemmas and tags. Said dashes may break 
            functionality in code that doesn't expect them, including 3rd party programs like
            ABLTagger and Nefnir. Moreover, anyone maintaining/expanding this code may not 
            be aware that Reynir does this. 
        So let's make sure the lemmas and the IFD tags are dash-free.
        """
        ifd_tag_list_full = [d.replace('-', '') for d in ifd_tag_list_with_dashes]
        lemma_list_spaces = [le.replace('-', '') for le in lemma_list_with_dashes]
        
        """
        Reynir also occasionally creates a single lemma out of more than one word, which can 
            lead to a number of problems (including immediate misalignment between the list of words 
            and the list of corresponding tags). So we check for empty spaces in each lemma and 
            split it accordingly, ensuring we always end up with a list of single-word lemmas.
        """
        lemma_list = [space_split_words for unsplit_entry in lemma_list_spaces for space_split_words in unsplit_entry.split(" ")]
        ifd_tag_list = [c[:1] for c in ifd_tag_list_full]
        
        for i in range(0, number_of_tags):
            word_tuple = (lemma_list[i], ifd_tag_list[i])
            lemmatized_words.append(word_tuple)
            
    return lemmatized_words


def text_tag(tok_text, model_type):
    """
    ABLTagger strongly prefers that each phrase end with some kind of punctuation.
        Before we pass it our input, we check for a full stop or a question mark;
        if neither is present, we add the former.
    """
    with open ("abl/deepmark.txt", "w+", encoding="utf-8") as file_out:
        for tokenized_phrase in tok_text:
            last_token = str(tokenized_phrase[-1])
            if not ( (last_token == ".") or (last_token == "?") ):
                tokenized_phrase.append(".")
            for token in tokenized_phrase:
                file_out.write(str(token))
                file_out.write('\n')
    file_out.close()

    subprocess.call(["python3", "tag.py", "--input", "deepmark.txt", "--model", model_type], cwd="abl")

    return "abl/deepmark.txt.tagged"

   
def text_lemmatize(file_tokenized):
    with open (file_tokenized, "r", encoding="utf-8") as abl_extra_lines:
        with open ("Nefnir/abl_output.txt", "w", encoding="utf-8") as abl_out:
            for line in abl_extra_lines:
                str_line_unstripped = str(line)
                str_line = str_line_unstripped.rstrip()
                if (str_line.startswith('.') or str_line.startswith('?')):
                    abl_out.write(str_line + "\n\n")
                elif (str_line != ""):
                    abl_out.write(str_line + "\n")
    abl_extra_lines.close()
   
    subprocess.call(["python3", "nefnir.py", "-i", "abl_output.txt", "-o", "lemmas.txt"], cwd="Nefnir")

    lemmatized_lists = []
    current_list_of_tuples = []
    with open ("Nefnir/lemmas.txt", "r", encoding="utf-8") as nefnir_lemmas:
        for n_line in nefnir_lemmas:
            str_n_unstripped= str(n_line)
            str_n = str_n_unstripped.rstrip()
            """
            Nefnir divides on blank spaces, so if the line is either blank or
                starts with something that's not alphanumeric, we've reached
                the end of our phrase and can add it to lemmatized_lists.
            """
            str_starting_character = str_n[:1]
            #De Morgan's laws again: not A or not B <==> not (A and B)
            if not ( (str_n) and (str_starting_character.isalnum()) ):
                if (len(current_list_of_tuples) > 0):
                    lemmatized_lists.append(current_list_of_tuples)
                current_list_of_tuples = []
            else:
                line_entries = str_n.split("\t")
                str_word = line_entries[2]
                str_category_full_ifd = line_entries[1]
                str_category_first_char = str_category_full_ifd[0]
                word_tuple = (str_word, str_category_first_char)
                current_list_of_tuples.append(word_tuple)
    nefnir_lemmas.close()
    
    return lemmatized_lists
	

def add_candidate_to_global_list(candidate_string):
    global term_candidate_list
    
    term_wordcount = len(candidate_string.split())
    term_already_exists = False

    for existing_entry in term_candidate_list:
        if existing_entry[0] == candidate_string:
            term_already_exists = True
            existing_entry[1] += 1
            break
    if not term_already_exists:
        term_candidate_list.append([candidate_string, 1, 0, 0, term_wordcount, 0.0, 0, -1.0])


def check_for_stopwords(candidate_string):
    global stop_list
    found_stopword = False
    
    for stop_string in stop_list:
        stop_string_regex = "(^|\s)" + stop_string + "(\s|\.|$)"
        stop_pattern = re.compile(stop_string_regex, re.IGNORECASE)
        
        if( stop_pattern.search(candidate_string) is not None ):
            found_stopword = True
            return found_stopword
        
    return found_stopword


def parse(lemmatized_line):
    global term_candidate_list
    global pattern_list
    global stop_list

    number_of_words_in_sentence = len(lemmatized_line)

    #Starting at each successive word in our candidate sentence...
    for sentence_word_index, current_word in enumerate(lemmatized_line):
        #...go through every category pattern that's sufficiently short...
        for pattern_index, pattern_type in enumerate(pattern_list):
            if( len(pattern_type) + sentence_word_index <= number_of_words_in_sentence ):
                match = True
                candidate_sentence = []
                pattern_range = len(pattern_type)
                #...and compare side-by-side the sequence of pattern tags and word tags.
                for category_index, category_type in enumerate(pattern_type):
                    if( category_type != lemmatized_line[sentence_word_index+category_index][1] ):
                        """
                        If we spot a mismatch, immediately stop checking this particular pattern,
                        break the innermost "for" loop, and begin checking the next pattern.
                        """
                        match = False
                        break
                    else:
                        """
                        If there's a match between the pattern tag and the word tag at this particular offset,
                        add that one word to candidate_sentence[] and check the next word in line.
                        """
                        candidate_sentence.append(lemmatized_line[sentence_word_index+category_index][0])
                if(match):
                    """
                    We've completed all comparisons for this particular pattern at this particular
                    offset in our candidate, and we've found a match. Convert candidate_sentence to
                    a string, check it's free of any stoplist phrases and, if so, add it to our 
                    global list of candidates.
                    Note that no matter whether this particular pattern occurred in the sentence,
                    we'll keep checking all other patterns from the *same* starting point in that
                    sentence *before* we move our starting point to the sentence's next word in line.
                    As a result, we're counting all candidate occurrences, including nested ones.
                    """
                    sentence_string = " ".join(candidate_sentence)
                    if not check_for_stopwords(sentence_string):
                        add_candidate_to_global_list(sentence_string)
                        
                        
def calculate_c_values():
    global term_candidate_list
    
    wordcount_index = 4
    term_text_index = 0
    c_value_index = 5
    
    term_candidate_list.sort(key=lambda x: x[4], reverse=True)

    start = 0
    max_index_number = len(term_candidate_list) - 1
    highest_wordcount = term_candidate_list[start][wordcount_index]
    
    while (start <= max_index_number) and (term_candidate_list[start][wordcount_index] >= highest_wordcount):
        start += 1

    """
    i and j are list indices used to repeatedly scan down the candidate
    list as we search for smaller terms nested inside larger ones.
    If every candidate in the entire list has the same number of words,
    the program will automatically skip the "range" for-loop below.
    """
    i = start
    
    for j in range (start, max_index_number+1):
        if(term_candidate_list[j][wordcount_index] < term_candidate_list[i][wordcount_index]):
            i = j
        for larger_term in range(0, i):
            if term_candidate_list[j][term_text_index] in term_candidate_list[larger_term][term_text_index]:
                """
                Index 2 is the sum of non-nested occurrences of every larger term that
                contains j a subterm (i.e. each larger term's total frequency minus its 
                frequency specifically as a subterm of some even *larger* term).
                """
                term_candidate_list[j][2] += (term_candidate_list[larger_term][1] - term_candidate_list[larger_term][2])
                #Index 3 is the number of *unique* larger terms of which j is a subterm.
                term_candidate_list[j][3] += 1

    for term in term_candidate_list:
    
        log2a = math.log(term[wordcount_index], 2.0)
        constant_i = 1.0
        small_c = constant_i + log2a
        f_a = term[1]
        SUM_bTa_f_b = term[2]
        P_Ta = term[3]

        if (term[wordcount_index] == highest_wordcount) or (P_Ta < 1):
            term[c_value_index] = small_c * f_a
        else:
            term[c_value_index] = small_c * (f_a - ((1.0/P_Ta) * SUM_bTa_f_b))


def find_levenshtein_distances():
    global known_term_list
    global term_candidate_list

    number_of_known_terms = len(known_term_list)
    if( number_of_known_terms > 0 ):
        for t in term_candidate_list:
            lowest_distance = 1000
            for k in range(1, number_of_known_terms):

                """
				1) mode="NW" means the candidate must be an exact match for a known term.
				We do have an option ("HW") for substring searches, but that would lead to false positives.
				2) task="distance" avoids wasting time trying to chart an optimal L-path (which we're
				not looking for anyway).
				"""
                curr_lev_comparison = edlib.align(t[0], known_term_list[k], mode="NW", task="distance")
                curr_lev_distance = curr_lev_comparison["editDistance"]
                if( curr_lev_distance < lowest_distance):
                    lowest_distance = curr_lev_distance
            t[6] = lowest_distance


def find_common_roots():
    global known_term_list_roots 
    global term_candidate_list
    
    resources = {
        "modifiers": os.path.join(os.path.dirname(__file__), 'resources', 'modifiers.dawg'),
        "heads": os.path.join(os.path.dirname(__file__), 'resources', 'heads.dawg'),
        "templates": os.path.join(os.path.dirname(__file__), 'resources', 'templates.dawg'),
        "splits": os.path.join(os.path.dirname(__file__), 'resources', 'splits.dawg')
    }
    kv = kvistur.Kvistur(**resources)
    
    for candidate_line in term_candidate_list:
        number_of_compound_words = 0
        stem_match_counter = 0
        match_ratio = 0.0
        candidate_word_list = candidate_line[0].split()
        for candidate_word in candidate_word_list:
            score, tree = kv.decompound(candidate_word)
            candidate_compound_list = []
            candidate_compound_list = tree.get_atoms()
            if( len(candidate_compound_list) > 1):
                number_of_compound_words += 1
                candidate_last_stem = candidate_compound_list[-1]
                for known_roots_line in known_term_list_roots:
                    for segmented_word in known_roots_line:
                        if(candidate_last_stem == segmented_word[-1]):
                            stem_match_counter += 1
        
        if( number_of_compound_words < 1 ):
            match_ratio = -1.0
        else:
            match_ratio = stem_match_counter/number_of_compound_words
        
        candidate_line[7] = match_ratio
                



def filter_results(c_value_threshold, l_distance_threshold=None, stem_ratio_threshold=None):
    global term_candidate_list
    global known_term_list

    filtered_terms = []
    extra_thresholds = False

    if( (l_distance_threshold is not None) and (stem_ratio_threshold is not None) ):
        extra_thresholds = True
    
    for t in term_candidate_list:

        """
        First, let's eliminate any candidates that already exist in known_term_list.
        (We don't want do do this earlier in the program because these candidates
        may contain new and unknown *nested* terms, and we've a better chance of
        finding those in the program's statistical calculations if we haven't yet
        eliminated anything.)
        There are several ways to implement this particular check, some faster than
        others. The lists of term candidates and known terms aren't likely to be
        long enough to affect performance, but if that changes, using "set" or "bisect"
        instead of "in", and pre-alphabetizing the list of known terms, might help.
        """
        if( extra_thresholds and (t[0] in known_term_list) ):
            #Candidate is a known term, so we won't add it to our filtered list.
            continue
        
        current_term = []
        passed_c = False
        passed_l = False
        s_exists = False
        passed_s = False
        
        if( t[5]>=c_value_threshold ):
            passed_c = True

        if(extra_thresholds):
            if( t[6]<=l_distance_threshold ):
                    passed_l = True
            if( t[7] >= 0.0 ):
                s_exists = True
                if (t[7] >= stem_ratio_threshold ):
                    passed_s = True
        
        if( (passed_c) and (not extra_thresholds) ):
            current_term.append(t[0])
            current_term.append(t[5])
            filtered_terms.append(current_term)
        elif( (passed_c) or ((extra_thresholds) and ((passed_l) or ((s_exists) and (passed_s))))):
            current_term.append(t[0])
            current_term.append(t[5])
            current_term.append(t[6])
            current_term.append(t[7])
            filtered_terms.append(current_term)
    
    return filtered_terms


def output_results(file_output, result_list, term_file_included):
    with open (file_output, "w", encoding="utf-8") as file_out:
        for item in result_list:
            str_out = ""
            last_item_index = len(item) - 1
            """
            Note: The number of terms is likely not that high. However, if it grows to
            a thoroughly nontrivial size, using .join() here instead if += would improve 
            runtime and decrease memory use (at the cost of readability).
            """
            for idx, val in enumerate(item):
                str_out += str(val)
                if(idx < last_item_index):
                    str_out += "\t"
                else:
                    str_out += "\n"
            file_out.write(str_out)
    file_out.close()

#
#
##### Start of functions used only for internal testing. May be safely ignored.
"""
def lemmatize_gold_standard(r):
    lemmatized_entries = []

    with open(FILE_GOLDSTANDARD_TERMS, encoding="utf-8") as file_in_gold:
        with open (FILE_GOLDSTANDARD_LEMMATIZED, "w", encoding="utf-8") as file_out_gold:
            for gold_in_line in iter(file_in_gold.readline, ''):
                    gold_s = r.parse_single(gold_in_line)
                    if gold_s.tree is not None:
                        #New code
                        lemmatized_gold_string_dashes = gold_s.tree.lemma
                        lemmatized_gold_string = re.sub('-', '', lemmatized_gold_string_dashes)
                        #Older code, use if newer code breaks
                        #lemmatized_gold_line_dashes = gold_s.tree.lemmas
                        #lemmatized_gold_line = [g.replace('-', '') for g in lemmatized_gold_line_dashes]
                        #lemmatized_gold_string = ' '.join(str(x) for x in lemmatized_gold_line)
                        lemmatized_entries.append(lemmatized_gold_string)
                        file_out_gold.write(lemmatized_gold_string + "\n")
        file_out_gold.close()
    file_in_gold.close()

    return lemmatized_entries


def load_lemmatized_gold_standard():
    lemmatized_entries = []
    with open(FILE_GOLDSTANDARD_LEMMATIZED, encoding="utf-8") as file_in_gold:
        for gold_in_line in iter(file_in_gold.readline, ''):
            gold_line_stripped = gold_in_line.rstrip()
            lemmatized_entries.append(gold_line_stripped)
    file_in_gold.close()
    return lemmatized_entries


def validate_results(known_terms_included, c_value_threshold, l_distance_threshold, stem_ratio_threshold, filtered_list):
    #Function to check how many terms from gold_standard_list I managed to find at all 
    # in term_candidate_list, how many made it through to filtered_list, what C/L/S values
    # they have, and which ones need to change (once while running Reynir; and once while
    # running ABLTagger & Nefnir) in order for the terms to all be found and make it from 
    # term_candidate_list to filtered_list

    global term_candidate_list
    global gold_standard_list

    #We want to see how high we can set the C and stem thresholds that a
    # candidate has to meet, and how low we can set the L threshold it also
    # has to meet, while still maintaining as close to a 100% recall as possible
    # (and thus hopefully improving precision somewhat as well, though that's
    # something we honestly can't measure).
    #c_value_threshold = 3.0
    #l_distance_threshold = 10
    #Temporary values while I test compound matching
    #TODO: RENAME these (or toss out) so I don't simply overwrite the values that were passed in
    c_value_threshold = 3.0
    l_distance_threshold = 10
    stem_ratio_threshold = 0.0
    filtered_terms = []
    
    #TODO: Look into changing the global FILE_OUT_RESULTS somehow
    with open (FILE_OUT_RESULTS, "w", encoding="utf-8") as file_out_results:
        for g in gold_standard_list:
            found_g = False
            passed_c = False
            passed_l = False
            passed_s = False
            for t in term_candidate_list:
                
                #TODO: Use the wordcount as a sliding scale so we can apply
                #the number of test matches. 
                #OR: Use the count of compounds against the number of matches
                t_wordcount = t[4]
                
                
                if( g == t[0]):
                #Let's play with C-values and L-distances to find a nice threshold
                #if( (g == t[0]) and (t[5]>3.0) and (t[6]<10)):
                    found_g = True
                    
                    if (t[5]>c_value_threshold):
                        passed_c = True
                    if (t[6]<l_distance_threshold):
                        passed_l = True
                    if passed_c and passed_l:
                        file_out_results.write("\"" + str(t[0]) + "\": C-value " + str(t[5]) + ", L-distance " + str(t[6]) + ",match ratio " + str(t[7]) + "\n")
                    elif (not passed_c) and (passed_l):
                        file_out_results.write("\"" + str(t[0]) + "\": C-value FAIL at " + str(t[5]) + ", but L-distance pass at " + str(t[6]) + "\n")
                    elif (passed_c) and (not passed_l):
                        file_out_results.write("\"" + str(t[0]) + "\": C-value pass at " + str(t[5]) + ", but L-distance FAIL at " + str(t[6]) + "\n")
                    else:
                        file_out_results.write("\"" + str(t[0]) + "\" BOTH FAILED: C-value " + str(t[5]) + " and L-distance " + str(t[6])  + "\n")
                    #ATH: Þetta break mun líklega stoppa "for t in term_candidate_list". Nota "continue"?
                    #break
                    continue
            if not found_g:
                file_out_results.write("Did not find \"" + str(g) + "\"\n")
    file_out_results.close()
"""
##### End of functions used for internal testing.
#
#
    
 
def main():

    r = Reynir()
    file_input = ""
    file_output = ""
    file_known_terms_lemmas = ""
    file_known_terms_all = ""
    known_terms_exist = False
    file_stoplist = ""
    
    """
    use_reynir determines whether we employ Reynir for the entire grammatical section. If not, we
    still use it for minor tasks, but rely on ABL for tagging and Nefnir for lemmatization.
    """
    use_reynir = False
    threshold_c_value = DEFAULT_C_VALUE
    threshold_l_distance = DEFAULT_L_DISTANCE
    threshold_s_ratio = DEFAULT_S_RATIO
    result_list = []
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", help="Name of (UTF-8) file containing the program's input", required=True)
    parser.add_argument("-o", "--output_file", help="Name of (UTF-8) file to which program will write its output", required=True)
    parser.add_argument("-g", "--use_reynir", help="Toggles use of Reynir for grammatical parse; otherwise, ABL and Nefnir are used", default=False)
    parser.add_argument("-c", "--c_value", help="Override the default (" + str(DEFAULT_C_VALUE) + ") C-value", default=DEFAULT_C_VALUE)
    parser.add_argument("-l", "--l_distance", help="Override the default (" + str(DEFAULT_L_DISTANCE) + ") Levenshtein distance", default=DEFAULT_L_DISTANCE)
    parser.add_argument("-s", "--s_ratio", help="Override the default (" + str(DEFAULT_S_RATIO) + ") root ratio", default=DEFAULT_S_RATIO)
    parser.add_argument("-t", "--stoplist_file", help="Optional (UTF-8) file containing a stop-list in lemmatized form")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-f", "--knownterms_lemma_file", help="Optional (UTF-8) file containing known terms in lemmatized form")
    group.add_argument("-a", "--knownterms_all_file", help="Optional (UTF-8) input file containing known terms in standard form, then in lemmatized form, then their IFD tag sequences")
    
    try:
        args = parser.parse_args()
    except:
        sys.exit(0)
    
    file_input = args.input_file
    file_output = args.output_file
    use_reynir = args.use_reynir
    """
    We'll be a little forgiving with types here, and cast str to float or int rather than exiting.
    However, any threshold values still invalid after casting are replaced by the default ones.
    """
    threshold_c_value = float(args.c_value)
    if( threshold_c_value < 0.0 ):
        threshold_c_value = DEFAULT_C_VALUE
    threshold_l_distance = int(args.l_distance)
    if( threshold_l_distance < 0 ):
        threshold_l_distance = DEFAULT_L_DISTANCE
    threshold_s_ratio = float(args.s_ratio)
    if( threshold_s_ratio < 0.0 ):
        threshold_s_ratio = DEFAULT_S_RATIO

    file_stoplist = args.stoplist_file

    if( args.knownterms_lemma_file ):
        file_known_terms_lemmas = args.knownterms_lemma_file
    elif( args.knownterms_all_file ):
        file_known_terms_all = args.knownterms_all_file
    else:
        pass

    load_known_terms(r, file_known_terms_lemmas, file_known_terms_all)
    if( len(known_term_list)>0 ):
        known_terms_exist = True
    
    global pattern_list
    pattern_list = populate_pattern_list(r, DEFAULT_PATTERN_FILE)

    global known_term_list_roots
    known_term_list_roots = load_roots_from_known_terms()
    
    global stop_list
    stop_list = load_stop_list(file_stoplist)

    """
    #Internal: Test code begins
    #Note: Use the lemmatize_gold_standard() function ONLY to create a brand new list. The 
    # outcome will contain numerous lemmatization errors that have to be fixed by hand.
    #XX#gold_standard_list = lemmatize_gold_standard(r)
    global gold_standard_list
    gold_standard_list = load_lemmatized_gold_standard()
    time_start = datetime.now()
    time_curr = datetime.now()
    #Internal: Test code ends
    """

    with open(file_input, "r", encoding="utf-8") as file_in:

        if( use_reynir ):
            for newline in iter(file_in.readline, ''):
                    tokenized_line = line_tokenize(newline)
                    pos_tagged_line = line_tag(tokenized_line, r)
                    lemmatized_line = line_lemmatize(pos_tagged_line)
                    parse(lemmatized_line)
        else:
            tokenized_text = []
            lemmatized_text = []
            for newline in iter(file_in.readline, ''):
                tokenized_line = line_tokenize(newline)
                tokenized_text.append(tokenized_line)
        
            """
            Note that this function call effectively hardcodes use of the "Light" model.
            We can also use the far more detailed "Full" model, but that requires 16 GB 
            of RAM and will obviously take longer at runtime.
            """
            abl_tagged_file = text_tag(tokenized_text, "Light")
            lemmatized_text = text_lemmatize(abl_tagged_file)
            for item in lemmatized_text:
                parse(item)
        file_in.close()
        
        calculate_c_values()
        if( known_terms_exist ):
            find_levenshtein_distances()
            find_common_roots()

        if( known_terms_exist ):
            result_list = filter_results(threshold_c_value, threshold_l_distance, threshold_s_ratio)
        else:
            result_list = filter_results(threshold_c_value)

        output_results(file_output, result_list, known_terms_exist)

    """
    #Internal code for gold standard validation and for speed-testing
    validate_results(known_terms_exist, threshold_c_value, threshold_l_distance, threshold_s_ratio, result_list)
    time_end = datetime.now()
    time_total = time_end - time_start
    print("Time taken in seconds: " + str(time_total.total_seconds()))
    #Internal code ends
    """


if __name__ == '__main__':
    main()