import os

import nltk
import nltk.data
import tqdm


def combine_wiki_files(wiki_path, output_file="en_wiki.txt"):
    data_path = os.path.join(wiki_path, "wiki_docs")
    output_path = os.path.join(wiki_path, output_file)

    with open(output_path, "w") as output:
        for folder in tqdm.tqdm(sorted(os.listdir(data_path)), desc="Processing folders"):
            for file in sorted(os.listdir(os.path.join(data_path, folder))):
                f = open(os.path.join(data_path, folder, file), "r")
                output.writelines(f.readlines())
                f.close()


def is_doc_start(line):
    return line.startswith("<doc id")


def is_doc_end(line):
    return line.startswith("</doc>")


def tokenise_sentences(wiki_path):
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    fin_path = os.path.join(wiki_path, "en_wiki.txt")
    fout_path = os.path.join(wiki_path, "en_wiki_out.txt")

    with open(fin_path, "r") as fin, open(fout_path, "w") as fout:
        doc_start = False
        for line in tqdm.tqdm(fin, desc="Processing lines"):
            line = line.strip("\n")
            if not line:
                continue
            if is_doc_start(line):
                doc_start = True
                fout.write(f"{line}\n")
                continue
            if doc_start:
                doc_start = False
                continue
            if is_doc_end(line):
                fout.write(f"{line}\n\n")
                continue
            line = line.replace("[", "").replace("]", "").replace("()", "")
            sentences = "\n".join(sent_detector.tokenize(line))
            fout.write(sentences)


if __name__ == '__main__':
    wiki_dir = "/home/yumouwei/wikipedia"
    tokenise_sentences(wiki_dir)
