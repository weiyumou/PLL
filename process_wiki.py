import glob
import os
import re
import shutil

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


def tokenise_sentences(wiki_path, output_dir="en_wiki", train_perct=0.9):
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    fin_path = os.path.join(wiki_path, "en_wiki.txt")
    output_path = os.path.join(wiki_path, output_dir)
    os.makedirs(output_path, exist_ok=True)
    fout, doc_count = None, 0

    with open(fin_path, "r") as fin:
        doc_start = False
        for line in tqdm.tqdm(fin, desc="Processing lines"):
            line = line.strip("\n")
            if not line:
                continue
            if is_doc_start(line):
                doc_start = True
                art_id = re.search(r"(?<=id=\")[0-9]+(?=\")", line).group(0)
                fout = open(os.path.join(output_path, f"{art_id}.txt"), "w")
                continue
            if doc_start:
                doc_start = False
                continue
            if is_doc_end(line):
                fout.close()
                doc_count += 1
                continue

            sentences = "\n".join(sent_detector.tokenize(line))
            fout.write(sentences)

    train_path = os.path.join(output_path, "train")
    val_path = os.path.join(output_path, "val")
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    for idx, file in tqdm.tqdm(enumerate(glob.iglob(os.path.join(output_path, "*.txt"))), desc="Splitting"):
        if idx < int(doc_count * train_perct):
            shutil.move(file, train_path)
        else:
            shutil.move(file, val_path)


if __name__ == '__main__':
    wiki_dir = "/home/yumouwei/wikipedia"
    tokenise_sentences(wiki_dir)
