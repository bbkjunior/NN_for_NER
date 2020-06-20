from config import Config
import codecs
import re

from torch.utils.data import Dataset,DataLoader
import torch

START_TAG = '<START>'
STOP_TAG = '<STOP>'

def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    """
    return re.sub('\d', '0', s)

def load_sentences(path):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    for line in codecs.open(path, 'r', 'utf8'):
        line = zero_digits(line.rstrip())
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            word = line.split()
            assert len(word) >= 2
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences

def create_dico(item_list):
    """
    Create a dictionary of items from a list of list of items.
    """
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico

def create_mapping(dico, add_pad = False):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    if add_pad == True:
        sorted_items.insert(0,("<PAD>",None))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item

def word_mapping(sentences, add_padding = False):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """
    words = [[x[0].lower()  for x in s] for s in sentences]
    dico = create_dico(words)
    dico['<UNK>'] = 10000000 #UNK tag for unknown words
    word_to_id, id_to_word = create_mapping(dico, add_padding)
    print("Found %i unique words (%i in total)" % (
        len(dico), sum(len(x) for x in words)
    ))
    return dico, word_to_id, id_to_word

def char_mapping(sentences, add_padding = False):
    """
    Create a dictionary and mapping of characters, sorted by frequency.
    """
    chars = ["".join([w[0] for w in s]) for s in sentences]
    dico = create_dico(chars)
    char_to_id, id_to_char = create_mapping(dico, add_padding)
    print("Found %i unique characters" % len(dico))
    return dico, char_to_id, id_to_char

def tag_mapping(sentences, add_padding = False):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    tags = [[word[-1] for word in s] for s in sentences]
    dico = create_dico(tags)
    dico[START_TAG] = -1
    dico[STOP_TAG] = -2
    tag_to_id, id_to_tag = create_mapping(dico, add_padding)
    print("Found %i unique named entity tags" % len(dico))
    return dico, tag_to_id, id_to_tag

def lower_case(x,lower=False):
    if lower:
        return x.lower()
    else:
        return x

# ELEMENTWISE APPROACH
def prepare_dataset(sentences, word_to_id, char_to_id, tag_to_id, lower=False):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - word char indexes
        - tag indexes
    """
    data = []
    for s in sentences:
        str_words = [w[0] for w in s]
        words = [word_to_id[lower_case(w,lower) if lower_case(w,lower) in word_to_id else '<UNK>']
                 for w in str_words]
        # Skip characters that are not in the training set
        chars = [[char_to_id[c] for c in w if c in char_to_id]
                 for w in str_words]
        tags = [tag_to_id[w[-1]] for w in s]
        data.append({
            'str_words': str_words,
            'words': words,
            'chars': chars,
            'tags': tags,
        })
    return data

def load_data():


    config = Config.from_json_file('../config.json').to_dict()
    train_sentences = load_sentences(config['train_path'])
    test_sentences = load_sentences(config['test_path'])
    dev_sentences = load_sentences(config['dev_path'])

    dico_words, word_to_id, id_to_word = word_mapping(train_sentences)
    dico_chars, char_to_id, id_to_char = char_mapping(train_sentences)
    dico_tags, tag_to_id, id_to_tag = tag_mapping(train_sentences)

    train_data = prepare_dataset(
        train_sentences, word_to_id, char_to_id, tag_to_id)
    dev_data = prepare_dataset(
        dev_sentences, word_to_id, char_to_id, tag_to_id)
    test_data = prepare_dataset(
        test_sentences, word_to_id, char_to_id, tag_to_id)

    return train_data,dev_data,test_data,word_to_id, char_to_id, tag_to_id

# BATCH APPROACH
class NERDataset(Dataset):
    def __init__(self, conll_sentence_list, word_to_id, char_to_id, tag_to_id, max_word_len=64,
                 max_char_len=24, pad_index=0):
        self.words_list = []
        self.chars_list = []
        self.tags_list = []
        self.max_word_len = max_word_len
        self.max_char_len = max_char_len
        self.pad_index = pad_index

        self.word_to_id = word_to_id
        self.char_to_id = char_to_id
        self.tag_to_id = tag_to_id

        self.preprocess(conll_sentence_list)

    def __len__(self):
        return len(self.words_list)

    def padding(self, tokens_list, max_len):
        tokens_list = tokens_list[:max_len]

        tokens_list = tokens_list + [self.pad_index] * (max_len - len(tokens_list))
        assert len(tokens_list) == max_len

        return tokens_list

    def pad_chars_list(self, ch_list):
        pad_char_list = [self.pad_index] * self.max_char_len

        ch_list = ch_list[:self.max_word_len]

        for _ in range(len(ch_list), self.max_word_len):
            ch_list.append(pad_char_list)

        return ch_list

    def preprocess(self, conll_sentence_list):
        for s in conll_sentence_list:
            str_words = [w[0] for w in s]
            words = [self.word_to_id[lower_case(w) if lower_case(w) in self.word_to_id else '<UNK>']
                     for w in str_words]
            words = self.padding(words, self.max_word_len)
            self.words_list.append(torch.tensor(words))

            tags = [self.tag_to_id[w[-1]] for w in s]
            tags = self.padding(tags, self.max_word_len)
            self.tags_list.append(torch.tensor(tags))

            chars = [self.padding([self.char_to_id[c] for c in w if c in self.char_to_id], self.max_char_len)
                     for w in str_words]
            chars = self.pad_chars_list(chars)

            chars = torch.tensor(chars)
            self.chars_list.append(chars)

    #             break

    def __getitem__(self, index):
        return self.words_list[index], self.chars_list[index], self.tags_list[index]


def create_loader(batch_size = 2):
    config = Config.from_json_file('../config.json').to_dict()
    train_sentences = load_sentences(config['train_path'])
    test_sentences = load_sentences(config['test_path'])
    dev_sentences = load_sentences(config['dev_path'])

    dico_words, word_to_id, id_to_word = word_mapping(train_sentences, add_padding = True)
    dico_chars, char_to_id, id_to_char = char_mapping(train_sentences, add_padding = True)
    dico_tags, tag_to_id, id_to_tag = tag_mapping(train_sentences, add_padding = True)

    dataset_train = NERDataset(train_sentences, word_to_id, char_to_id, tag_to_id)
    dataset_valid = NERDataset(test_sentences, word_to_id, char_to_id, tag_to_id)
    dataset_test = NERDataset(dev_sentences, word_to_id, char_to_id, tag_to_id)

    train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=dataset_valid, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=True)
    for _ in train_loader:
        break
    for _ in valid_loader:
        break
    for _ in test_loader:
        break
    # print("loaders created!")
    return train_loader, valid_loader, test_loader, word_to_id, char_to_id, tag_to_id

def cut_words_and_tags(sequence):
    lens = 64 - (sequence == 0).sum(1)
    lens_max = max(lens)
    return sequence[:,:lens_max+2], lens

def cut_char(sequence):
    pad_mask = sequence != 0
    word_max = pad_mask.sum(dim=1).max()
    char_max = pad_mask.sum(dim=-1).max()
    sequence = sequence[:, :word_max+2, :char_max]
    return sequence

if __name__ == '__main__':
    # load_data()
    create_loader()