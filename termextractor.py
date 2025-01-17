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
# nltk.download('punkt')
import argparse
from reynir import Reynir
from tokenizer import tokenize, TOK
import idord.kvistur
import requests

API_LOCATION = "http://malvinnsla.arnastofnun.is"


class TermExtractor():

    def __init__(self, file_known_terms, file_patterns, file_stoplist):
        self.file_patterns = file_patterns
        self.file_known_terms = file_known_terms
        self.file_stoplist = file_stoplist

        #Note: Removing the dual-file option for now.
        #self.file_known_terms_lemmas = ""
        #self.file_known_terms_all = ""

        self.c_value_threshold = 3.0
        self.l_distance_threshold = 15
        self.s_ratio_threshold = 1.5

        #Reynir is *probably* more reentrant if kept as an instance variable, given
        # that it's going to be continually used for sentence parsing and storage.
        self.r = Reynir()

        self.known_term_list = []
        self.pattern_list = {}
        self.known_term_list_roots = []
        self.term_candidate_list = []
        self.stop_list = []
        self.term_candidate_list = []

        self.load_known_terms()
        self.populate_pattern_list()
        self.load_roots_from_known_terms()
        self.load_stop_list()

    def load_known_terms(self):
        #Note: Removing the dual-file option for now.
        """
        def load_known_terms(self, file_lemmas, file_all):
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
        """
        if( self.file_known_terms ):
            with open(self.file_known_terms, "r", encoding="utf-8") as file_l:
                for newline_l in iter(file_l.readline, ''):
                    line_l_full = str(newline_l)
                    line_l_string = line_l_full.rstrip()
                    if( line_l_string ):
                        self.known_term_list.append(line_l_string)
            file_l.close()

    def populate_pattern_list(self):
        if( self.file_patterns ):
            self.pattern_list[tuple()] = True
            with open(self.file_patterns, "r", encoding="utf-8") as file_p:
                for line_p in file_p:
                    tag_seq = []
                    for tag in line_p.split():
                        # only uses first character of tag
                        tag_seq.append(tag[0])
                        # set this sequence's value to false because pattern is incomplete.
                        if tuple(tag_seq) not in self.pattern_list:
                            self.pattern_list[tuple(tag_seq)] = False
                    self.pattern_list[tuple(tag_seq)] = True
                    # line_p_list = tuple(ifd_tag[0] for ifd_tag in line_p_str.split())
                    # self.pattern_list.add(line_p_list)
            # print(self.pattern_list)
            file_p.close()

    def load_roots_from_known_terms(self):
        if( len(self.known_term_list)>0 ):
            resources = {
                "modifiers": os.path.join(os.path.dirname(__file__), 'resources', 'modifiers.dawg'),
                "heads": os.path.join(os.path.dirname(__file__), 'resources', 'heads.dawg'),
                "templates": os.path.join(os.path.dirname(__file__), 'resources', 'templates.dawg'),
                "splits": os.path.join(os.path.dirname(__file__), 'resources', 'splits.dawg')
            }
            kv = kvistur.Kvistur(**resources)

            for line in self.known_term_list:
                root_line = []
                line_list = line.split()
                for word in line_list:
                    score, tree = kv.decompound(word)
                    compound_list = []
                    compound_list = tree.get_atoms()
                    if( len(compound_list) > 1):
                        root_line.append(compound_list)
                if( len(root_line) > 0):
                    self.known_term_list_roots.append(root_line)

    def load_stop_list(self):
        if( self.file_stoplist ):
            with open(self.file_stoplist, "r", encoding="utf-8") as file_s:
                for newline_s in iter(file_s.readline, ''):
                    line_s_full = str(newline_s)
                    line_s_string = line_s_full.rstrip()
                    if( line_s_string ):
                        """
                        line_s_list = [word for word in line_s_string.split()]
                        stop_l.append(line_s_list)
                        """
                        self.stop_list.append(line_s_string)
            file_s.close()

    def line_tokenize(self, newline):
        list_of_tokenized_words = []

        for token in tokenize(newline):
            kind, txt, val = token
            if kind == TOK.WORD:
                list_of_tokenized_words.append(txt)
        return list_of_tokenized_words

    def line_tag(self, tokenized_list):
        str_tokens = " ".join(tokenized_list)
        parsed_tokens = self.r.parse_single(str_tokens)
        return parsed_tokens

    def line_lemmatize(self, pos_tagged_sentence):
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

    def text_tag(self, tok_text, model_type):
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

    def text_lemmatize(self, file_tokenized):
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

    @staticmethod
    def tag_and_lemmatize(text):
        """
        Input: A text string.
        Output: A list of lists of tuples containing lemmas, tags, and words
        """

        HYPHENS = "-–—" # HACK: hotfix because of stupid tokenizer normalisation
        DQUOTES = '"“„”«»' # HACK: hotfix because of stupid tokenizer normalisation
        res = requests.post(
            url=API_LOCATION,
            data={'text': text, 'model_type': 'coarse', 'lemma': 'on'})
        sentences = [sent for para in res.json()['paragraphs'] for sent in para['sentences']]
        # print(sentences)
        outputs = []
        sentence = []
        text_locations = []
        i = 0
        for sentence in sentences:
            output_sentence = []
            for word_obj in sentence:
                lemma, mark, ord = (word_obj['lemma'], word_obj['tag'], word_obj['word'])
                while not text[i:i+len(ord)] == ord:
                    # HACK: Same hack as above
                    if text[i:i+len(ord)] in HYPHENS and ord in HYPHENS:
                        break
                    if text[i:i+len(ord)] in DQUOTES and ord in DQUOTES:
                        break
                    i+=1
                    # print(text[i:i+len(ord)], ord)
                    if i>len(text):
                        break
                output_sentence.append((lemma, mark, ord, i))
                i+=len(ord)
            outputs.append(output_sentence)
        return outputs

    def add_candidate_to_global_list(self,
                                     candidate_string,
                                     unlemmatized_string,
                                     start_end,
                                     term_candidate_list=None):
        if term_candidate_list is None:
            term_candidate_list = self.term_candidate_list
        term_wordcount = len(candidate_string.split())
        term_already_exists = False

        for existing_entry in term_candidate_list:
            if existing_entry["lemmas"] == candidate_string:
                term_already_exists = True
                # existing_entry["original"][1] += 1
                existing_entry["frequency"] += 1
                existing_entry["occurences"].append(unlemmatized_string)
                existing_entry["boundaries"].append(start_end)
                break
        if not term_already_exists:
            term_candidate_list.append({
                "lemmas": candidate_string,
                "frequency": 1,
                "parent_count": 0,
                "parent_types": 0,
                "wordcount": term_wordcount,
                "c_value": 0.0,
                "distance": 0,
                "s_ratio": -1,
                "occurences": [(unlemmatized_string,)],
                "boundaries": [start_end]
            })

    def check_for_stopwords(self, candidate_string):
        found_stopword = False

        for stop_string in self.stop_list:
            stop_string_regex = "(^|\s)" + stop_string + "(\s|\.|$)"
            stop_pattern = re.compile(stop_string_regex, re.IGNORECASE)

            if(stop_pattern.search(candidate_string) is not None):
                found_stopword = True
                return found_stopword

        return found_stopword

    def parse(self, lemmatized_line, term_candidate_list=None):
        if term_candidate_list is None:
            term_candidate_list = self.term_candidate_list
        pl = self.pattern_list
        sentence_length = len(lemmatized_line)

        # O(n*(complexity of inner loop)) => O(n*O(1)) == O(n)
        # Improves upon earlier implementation in that complexity doesn't vary by number of patterns
        for i in range(len(lemmatized_line)):
            lemmas = []
            words = []
            tags = []
            start_index = lemmatized_line[i][3]
            tag_tuple = tuple()
            j = 0
            # Dict membership check is O(1)
            # tuple conversion is O(k) with k<length of longest pattern so O(1)
            while tag_tuple in pl and i+j < sentence_length:
                lemma, tag, word, idx = lemmatized_line[i+j]
                lemmas.append(lemma)
                words.append(word)
                tags.append(tag)
                tag_tuple = tuple(tags)
                if tag_tuple in pl and pl[tag_tuple]:
                    end_index = idx + len(word)
                    lemma_string = " ".join(lemmas)
                    word_string = " ".join(words)
                    start_end = [str(start_index), str(end_index)]
                    if not self.check_for_stopwords(lemma_string):
                        self.add_candidate_to_global_list(
                            lemma_string,
                            word_string,
                            start_end,
                            term_candidate_list)

                j+=1

        # #Starting at each successive word in our candidate sentence...
        # for sentence_word_index, current_word in enumerate(lemmatized_line):
        # O(k*n*m) where k is no. of patterns, n is length of sentence and m is length of longest pattern
        # m is generally low so ~O(k*n)
        #
        #     #...go through every category pattern that's sufficiently short...
        #     for pattern_index, pattern_type in enumerate(self.pattern_list):
        #         if(len(pattern_type) + sentence_word_index <= sentence_length):
        #             match = True
        #             candidate_sentence = []
        #             unlemmatized_words = []
        #             pattern_range = len(pattern_type)
        #             # ...and compare side-by-side the sequence of pattern tags and word tags.
        #             for category_index, category_type in enumerate(pattern_type):
        #                 if(category_type != lemmatized_line[sentence_word_index+category_index][1]):
        #                     """
        #                     If we spot a mismatch, immediately stop checking this particular pattern,
        #                     break the innermost "for" loop, and begin checking the next pattern.
        #                     """
        #                     match = False
        #                     break
        #                 else:
        #                     """
        #                     If there's a match between the pattern tag and the word tag at this particular offset,
        #                     add that one word to candidate_sentence[] and check the next word in line.
        #                     """
        #                     candidate_sentence.append(lemmatized_line[sentence_word_index+category_index][0])
        #                     unlemmatized_words.append(lemmatized_line[sentence_word_index+category_index][2])
        #             if(match):
        #                 """
        #                 We've completed all comparisons for this particular pattern at this particular
        #                 offset in our candidate, and we've found a match. Convert candidate_sentence to
        #                 a string, check it's free of any stoplist phrases and, if so, add it to our
        #                 global list of candidates.
        #                 Note that no matter whether this particular pattern occurred in the sentence,
        #                 we'll keep checking all other patterns from the *same* starting point in that
        #                 sentence *before* we move our starting point to the sentence's next word in line.
        #                 As a result, we're counting all candidate occurrences, including nested ones.
        #                 """
        #                 sentence_string = " ".join(candidate_sentence)
        #                 unlemmatized_phrase = " ".join(unlemmatized_words)
        #                 if not self.check_for_stopwords(sentence_string):
        #                     self.add_candidate_to_global_list(
        #                         sentence_string,
        #                         unlemmatized_phrase,
        #                         lemmatized_line,
        #                         term_candidate_list)

    def calculate_c_values(self, term_candidate_list=None):
        if term_candidate_list is None:
            term_candidate_list = self.term_candidate_list

        term_candidate_list.sort(key=lambda x: x["wordcount"], reverse=True)

        start = 0
        max_index_number = len(term_candidate_list) - 1
        highest_wordcount = term_candidate_list[start]["wordcount"]

        while (start <= max_index_number) and (term_candidate_list[start]["wordcount"] >= highest_wordcount):
            start += 1

        """
        i and j are list indices used to repeatedly scan down the candidate
        list as we search for smaller terms nested inside larger ones.
        If every candidate in the entire list has the same number of words,
        the program will automatically skip the "range" for-loop below.
        """
        i = start

        for j, term in enumerate(term_candidate_list[start:max_index_number+1]):
            if(term["wordcount"] < term_candidate_list[i]["wordcount"]):
                i = j
            for larger_term in term_candidate_list[0:i]:
                if term["lemmas"] in larger_term["lemmas"]:
                    """
                    Index 2 is the sum of non-nested occurrences of every larger term that
                    contains j a subterm (i.e. each larger term's total frequency minus its
                    frequency specifically as a subterm of some even *larger* term).
                    """
                    term["parent_count"] += (larger_term["frequency"] - larger_term["parent_count"])
                    #Index 3 is the number of *unique* larger terms of which j is a subterm.
                    term["parent_types"] += 1

        for term in term_candidate_list:

            log2a = math.log(term["wordcount"], 2.0)
            constant_i = 1.0
            small_c = constant_i + log2a
            f_a = term["frequency"]
            SUM_bTa_f_b = term["parent_count"]
            P_Ta = term["parent_types"]

            if (term["wordcount"] == highest_wordcount) or (P_Ta < 1):
                term["c_value"] = small_c * f_a
            else:
                term["c_value"] = small_c * (f_a - ((1.0/P_Ta) * SUM_bTa_f_b))

    def find_levenshtein_distances(self, term_candidate_list=None):
        if term_candidate_list is None:
            term_candidate_list = self.term_candidate_list

        if(len(self.known_term_list) > 0):
            for t in term_candidate_list:
                lowest_distance = 1000
                for known in self.known_term_list:

                    """
                    1) mode="NW" means the candidate must be an exact match for a known term.
                    We do have an option ("HW") for substring searches, but that would lead to false positives.
                    2) task="distance" avoids wasting time trying to chart an optimal L-path (which we're
                    not looking for anyway).
                    """
                    curr_lev_comparison = edlib.align(t["lemmas"], known, mode="NW", task="distance")
                    curr_lev_distance = curr_lev_comparison["editDistance"]
                    if(curr_lev_distance < lowest_distance):
                        lowest_distance = curr_lev_distance
                t["distance"] = lowest_distance

    def find_common_roots(self, term_candidate_list=None):
        if term_candidate_list is None:
            term_candidate_list = self.term_candidate_list
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
            candidate_word_list = candidate_line["lemmas"].split()
            for candidate_word in candidate_word_list:
                score, tree = kv.decompound(candidate_word)
                candidate_compound_list = []
                candidate_compound_list = tree.get_atoms()
                if(len(candidate_compound_list) > 1):
                    number_of_compound_words += 1
                    candidate_last_stem = candidate_compound_list[-1]
                    for known_roots_line in self.known_term_list_roots:
                        for segmented_word in known_roots_line:
                            if(candidate_last_stem == segmented_word[-1]):
                                stem_match_counter += 1

            if(number_of_compound_words < 1):
                match_ratio = -1.0
            else:
                match_ratio = stem_match_counter/number_of_compound_words

            candidate_line["s_ratio"] = match_ratio

    def filter_results(self, use_extra_thresholds=False, term_candidate_list=None):
        filtered_terms = []
        if term_candidate_list is None:
            term_candidate_list = self.term_candidate_list

        """
        if( (l_distance_threshold is not None) and (stem_ratio_threshold is not None) ):
            extra_thresholds = True
        """

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
            if( use_extra_thresholds and (t["lemmas"] in self.known_term_list) ):
                #Candidate is a known term, so we won't add it to our filtered list.
                continue

            passed_c = False
            passed_l = False
            s_exists = False
            passed_s = False

            if(t["c_value"]>=self.c_value_threshold):
                passed_c = True

            if(use_extra_thresholds):
                if( t["distance"]<=self.l_distance_threshold):
                    passed_l = True
                if( t["s_ratio"] >= 0.0 ):
                    s_exists = True
                    if (t["s_ratio"] >= self.s_ratio_threshold):
                        passed_s = True

            if((passed_c) and (not use_extra_thresholds)):
                # t["sentence"] = " ".join([x[2] for x in t["sentence"]])
                filtered_terms.append(t)
            elif((passed_c) or ((use_extra_thresholds) and ((passed_l) or (passed_s)))):
                # t["sentence"] = " ".join([x[2] for x in t["sentence"]])
                filtered_terms.append(t)

        return filtered_terms

    def convert_list_output(self, term_candidate_list=None):
        if term_candidate_list is None:
            term_candidate_list = self.term_candidate_list
        return [
            [
                x["lemmas"],
                x["frequency"],
                x["parent_count"],
                x["parent_types"],
                x["wordcount"],
                x["c_value"],
                x["distance"],
                x["s_ratio"]]
            for x in term_candidate_list
        ]

##### Class TermExtractor ends #####
