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
from termextractor import TermExtractor

def load_file_text(file_input):
    file_t = []
    with open(file_input, "r", encoding="utf-8") as file_in:
        for file_line in iter(file_in.readline, ''):
            newline_s_full = str(file_line)
            newline_s_clean = newline_s_full.rstrip()
            if( newline_s_clean ):
                file_t.append(newline_s_clean)
    file_in.close()   
    return file_t


def output_results(file_output, result_list):
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


def main():
    file_known_patterns = "default_patterns.txt"
    file_known_terms_lemmas = "baralemmur.txt"
    #Note: Removing the dual-file option for now.
    #file_known_terms_all = ""
    te = TermExtractor(file_known_terms_lemmas, file_known_patterns)

    #Hardcoded run options - change these as needed
    use_reynir = False
    known_terms_exist = True

    """
    Moving I/O to a separate function for improved readability 
    and (slightly better) decoupling
    """
    file_text = load_file_text("input.txt")
    
    if( use_reynir ):
        for newline in file_text:
            tokenized_line = te.line_tokenize(newline)
            pos_tagged_line = te.line_tag(tokenized_line)
            lemmatized_line = te.line_lemmatize(pos_tagged_line)
            te.parse(lemmatized_line)
    else:
        tokenized_text = []
        lemmatized_text = []
        for newline in file_text:
            tokenized_line = te.line_tokenize(newline)
            tokenized_text.append(tokenized_line)
    
        """
        Note that this function call effectively hardcodes use of the "Light" model.
        We can also use the far more detailed "Full" model, but that requires 16 GB 
        of RAM and will obviously take longer at runtime.
        """
        abl_tagged_file = te.text_tag(tokenized_text, "Light")
        lemmatized_text = te.text_lemmatize(abl_tagged_file)
        for item in lemmatized_text:
            te.parse(item)

    te.calculate_c_values()
    if( known_terms_exist ):
        te.find_levenshtein_distances()
        te.find_common_roots()

    """
    #Moving threshold values to the TermExtractor class itself, which means
    # this if-condition may no longer apply
    if( known_terms_exist ):
        result_list = re.filter_results(threshold_c_value, threshold_l_distance, threshold_s_ratio)

    else:
        result_list = re.filter_results(threshold_c_value)
    """
    result_list = te.filter_results(known_terms_exist)

    #output_results(file_output, result_list, known_terms_exist)
    output_results("output.txt", result_list)


if __name__ == '__main__':
    main()