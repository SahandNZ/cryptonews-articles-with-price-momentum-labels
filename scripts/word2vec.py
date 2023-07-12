import argparse
import os.path

import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch import optim

from definitions import DEEP_LEARNING_DIR, STATS_DIR
from src.utils.directory import create_directory_recursively
from src.word2vec.loss_fn import Word2VecLossFn
from src.word2vec.model import Model
from src.word2vec.utils import create_lookup_tables, train_skip_gram, load_words


def run_word2vec(args):
    words = load_words(args.textual_source, args.method, args.label)
    word2token, token2word = create_lookup_tables(words)
    tokens = list(token2word.keys())

    # load or create model
    root = os.path.join(DEEP_LEARNING_DIR, 'word2vec')
    path = os.path.join(root, 'model.pt')
    create_directory_recursively(root)
    if args.load_model and os.path.exists(path):
        model = torch.jit.load(path)
        model.eval()
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = Model(n_vocab=len(word2token), n_embed=args.embedding_size).to(device)
        criterion = Word2VecLossFn()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        train_skip_gram(tokens=tokens, model=model, criterion=criterion, optimizer=optimizer,
                        n_negative_samples=args.negative_sampling, batch_size=args.batch_size, n_epochs=args.epochs)

    # save model
    if args.save_model:
        model_scripted = torch.jit.script(model)
        model_scripted.save(path)

    if args.save_image:
        embeddings = model.in_embed.weight.to('cpu').data.numpy()
        tsne = TSNE()
        embeddings_tsne = tsne.fit_transform(embeddings[:args.visualizing_images, :])

        plt.figure(figsize=(10, 10))
        for i in range(len(embeddings_tsne)):
            plt.scatter(*embeddings_tsne[i, :], color='steelblue')
            plt.annotate(token2word[i], (embeddings_tsne[i, 0], embeddings_tsne[i, 1]), alpha=0.7)

        figure_path = os.path.join(STATS_DIR, 'word2vec.png')
        plt.xlabel("Words")
        plt.xticks([])
        plt.grid()
        plt.savefig(figure_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--textual-source', action='store', type=str, required=False, default='cryptonews')
    parser.add_argument('--labeling-method', action='store', type=str, required=False, default='color')
    parser.add_argument('--label', action='store', type=int, required=False, default=None)

    parser.add_argument('--load-model', action='store_true', required=False, default=False)
    parser.add_argument('--save-model', action='store_true', required=False, default=False)
    parser.add_argument('--save-image', action='store_true', required=False, default=False)

    parser.add_argument('--embedding-size', action='store', type=int, required=False, default=32)
    parser.add_argument('--negative-sampling', action='store', type=int, required=False, default=10)
    parser.add_argument('--epochs', action='store', type=int, required=False, default=1000)
    parser.add_argument('--batch-size', action='store', type=int, required=False, default=512)
    parser.add_argument('--visualizing-images', action='store', type=int, required=False, default=100)
    args = parser.parse_args()

    run_word2vec(args)


if __name__ == '__main__':
    main()
