
import os
import xml.etree.ElementTree as ET
import re
from pathlib import Path


class ProcessCorpus():
    def __init__(self, corpus_type="single_file", filepath=None, dirlist=None,
                 outputname="results.txt"):
        self.type = corpus_type
        self.results_path = os.path.join(
            Path(__file__).parent, f"processed/{outputname}.txt")

        if self.type == "single_file":
            assert(filepath is not None)
            self.process_single_file(filepath)

        elif self.type == "directories":
            assert(dirlist is not None)
            self.process_dirlist(dirlist)

    def custom_file_func(self, text):
        """If corpus_type is "single_file", this takes input file split into lines
           and outputs a list of strings, where each string is a document.
           If corpus_type is "directories", this takes input file split into lines
           and outputs a single string."""
        print("Not defined for parent class; specified in subclass")

    def process_single_file(self, fp):

        # Apply custom processing to get list of lines
        corpus_file = open(fp, 'r')
        lines = corpus_file.readlines()

        # get list of documents, split out from lines
        output = self.custom_file_func(lines)

        with open(self.results_path, "w") as f:
            for line in output:

                # Process numbers in line
                line = self._process_numbers(line)
                # Process punctuation in line
                line = self._process_punctuation(line)
                # remove extra space in line
                line = self._remove_extra_space(line)

                f.write(line)
                f.write("\n")

    
    def process_dirlist(self, dirlist):

        with open(self.results_path, "w") as res_f:
            # For file in dir in dirlist
            for d in dirlist:
                for fil in os.listdir(d):
                    # Apply custom processing
                    corpus_file = open(os.path.join(d, fil), 'r')
                    # Get individual document line from inputted file
                    lines = corpus_file.readlines()
                    # Here, custom processing yields file --> string
                    output = self.custom_file_func(lines)
                    # Process numbers
                    #output = self._process_numbers(output)
                    # Process punctuation
                    output = self._process_punctuation(output)
                    # Remove extra space
                    output = self._remove_extra_space(output)
                    if len(output) > 0:
                        res_f.write(output)
                        res_f.write("\n")

        res_f.close()


    def _process_numbers(self, t):

        t = t.replace("0", " zero ")
        t = t.replace("1", " one ")
        t = t.replace("2", " two ")
        t = t.replace("3", " three ")
        t = t.replace("4", " four ")
        t = t.replace("5", " five ")
        t = t.replace("6", " six ")
        t = t.replace("7", " seven ")
        t = t.replace("8", " eight ")
        t = t.replace("9", " nine ")

        return t


    def _remove_numbers(self, t):

        for i in range(10):
            t = t.replace(str(i), "")

        return t

    def _process_punctuation(self, t):

        punc =  [".", ",", "'", '"', ":", ";", "(", ")", "/", "^", "!", "?",
                "[", "]", "”", "“", "’", "‘", "_", "-", "}", "{", "*", "*",
                "|", "=", "$", "\\", "#", "⌉", "⌋", "⌊", "⌈", " _", "~", "‡", 
                "<", ">", "", "@", "+"]

        for x in punc:
            t = t.replace(x, "")
        return t


    def _remove_extra_space(self, t):
        while "  " in t:
            t = t.replace("  ", " ")
        return t