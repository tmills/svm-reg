#!/usr/bin/env python

import sys
from nltk import word_tokenize
from glob import glob
from os.path import basename, isdir, join
from os import makedirs

## Started writing this to do my own tokenization and pre-processing of the
## 20 newsgroups corpus, then learned that there are built in functions in
## scikit-learn to do this preprocessing. For now, to get started, I'll leave
## this alone and move over to doing that. Once I get going, I may find I
## want to do my own pre-processing, but hopefully sklearn's will be sufficient,
## since that will be more easily reportable/repeatable.

def main(args):
    if len(args) < 2:
        sys.stderr.write("This script reads in a directory with text files, and for every file\n")
        sys.stderr.write("in that directory, creates an equivalent file in the output directory\n")
        sys.stderr.write("containing the set of words with their counts.\n")
        sys.stderr.write("One required argument: <newsgroup directory> <output directory>\n")
        sys.exit(-1)

    if not isdir(args[1]):
        print("Creating output directory %s" % (args[1]))
        makedirs(args[1])

    for fn in glob(join(args[0], '*')):
        #print("Processing file %s" % fn)
        map = {}
        try:
            with open(fn) as x: txt = x.read()
            tokens = word_tokenize(txt.encode().lower())
            for token in tokens:
                if not map.has_key(token):
                    map[token] = 0

                map[token] += 1

            out_fn = join(args[1], basename(fn))
            out = open(out_fn, 'w')
            #print("Writing to file %s" % (out_fn))
            for key in map.keys():
                out.write("%s : %d\n" % (key, map[key]))

            out.close()
        except Exception as e:
            sys.stderr.write("Caught an error in file %s; skipping\n" % (fn))

if __name__ == "__main__":
    main(sys.argv[1:])
