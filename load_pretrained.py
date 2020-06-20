from config import Config
import codecs
import numpy as np

from prepocessing import load_data


def load_embeddings(word_to_id, test=False):
    config = Config.from_json_file('./config.json').to_dict()
    all_word_embeds = {}
    print("LOADING PREATRAINED ...")
    for i, line in enumerate(codecs.open("./data/glove.6B.100d.txt", 'r', 'utf-8')):
        s = line.strip().split()
        if len(s) == config['word_dim'] + 1:
            all_word_embeds[s[0]] = np.array([float(i) for i in s[1:]])
            if test == True:
                break

    # Intializing Word Embedding Matrix
    word_embeds = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (len(word_to_id), config['word_dim']))

    for w in word_to_id:
        if w in all_word_embeds:
            word_embeds[word_to_id[w]] = all_word_embeds[w]
        elif w.lower() in all_word_embeds:
            word_embeds[word_to_id[w]] = all_word_embeds[w.lower()]

    print('Loaded %i pretrained embeddings.' % len(all_word_embeds))

    return word_embeds


if __name__ == '__main__':
    _, _, _, word_to_id = load_data()
    emb = load_embeddings(word_to_id, test=True)