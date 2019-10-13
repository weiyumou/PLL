from torch.utils.data import Dataset
import spacy
from nltk.util import ngrams


class TextDataset(Dataset):
    def __init__(self, data, batch_size, TEXT) -> None:
        super(TextDataset, self).__init__()
        self.parse_sentences(data)
        # data = TEXT.numericalize([data.examples[0].text])
        # # Divide the dataset into bsz parts.
        # nbatch = data.size(0) // batch_size
        # # Trim off any extra elements that wouldn't cleanly fit (remainders).
        # data = data.narrow(0, 0, nbatch * batch_size)
        # # Evenly divide the data across the bsz batches.
        # self.data = data.reshape(nbatch, -1)

    def __getitem__(self, index: int):
        return super().__getitem__(index)

    def __len__(self) -> int:
        return self.data.size(0)

    @staticmethod
    def parse_sentences(data):
        paras = " ".join(data.examples[0].text).split("<eos>")
        for i, para in enumerate(paras[:100]):
            para = para.strip()
            if para and not TextDataset.is_header(para):
                nlp = spacy.load("en_core_web_sm")
                for sent in nlp(para).sents:
                    for item in ngrams(sent.text.split(), 9):
                        print(item)
                # print(i, para)

    @staticmethod
    def is_header(line):
        return line.startswith("=")
