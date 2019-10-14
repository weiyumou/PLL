import spacy
from nltk.util import ngrams
from torch.utils.data import Dataset
import tqdm


class TextDataset(Dataset):
    def __init__(self, text_data, ngram, TEXT) -> None:
        super(TextDataset, self).__init__()
        all_ngrams = self.parse_ngrams(text_data, ngram)
        self.data = TEXT.numericalize(all_ngrams).permute(1, 0)  # torch.Size([4294, 9])

    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self) -> int:
        return self.data.size(0)

    @staticmethod
    def parse_ngrams(data, ngram):
        paras = " ".join(data.examples[0].text).split("<eos>")
        # nlp = spacy.load("en_core_web_sm")
        all_ngrams = []
        for i, para in tqdm.tqdm(enumerate(paras), desc="Parsing Paragraphs", total=len(paras)):
            para = para.strip()
            if para and not para.startswith("="):
                for item in ngrams(para.split(), ngram):
                    all_ngrams.append(item)

                # # Segment sentences
                # for sent in nlp(para).sents:
                #     # Generate ngrams
                #     for item in ngrams(sent.text.split(), ngram):
                #         all_ngrams.append(item)
        return all_ngrams
