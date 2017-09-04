#!/usr/bin/env python

import sys
import glob

## WIP: most work moved over to classify_all_newsgroups_pairs.py
def main(args):
    if len(args) < 2:
        sys.stderr.write("Two required arguments: <Domain dir 1> <Domain dir 2>\n")
        sys.exit(-1)

    word2int = {}
    for fn in glob(args[0] + "/*"):
        with open(fn, 'r') as f:
            for line in f.lines():
                word, colon, count = line.split()
                if not word2int.has_key(word):
                    word2int[word] = len(word2int)


if __name__ == "__main__":
    main(sys.argv[1:])
