import argparse
import os.path

from transformers.models.bert_japanese.tokenization_bert_japanese import spm

from definitions import DEEP_LEARNING_DIR
from src.tokenizer.sp import load_corpus, load_spp, train_spp, cross_validation, calculate_unk_rate
from src.utils.directory import create_directory_recursively


def run_sentence_piece(args):
    corpus = load_corpus(args.textual_source, args.labeling_method)

    root = os.path.join(DEEP_LEARNING_DIR, 'tokenizer')
    path = os.path.join(root, 'sp.model')
    create_directory_recursively(root)
    if args.load_spp and os.path.exists(path):
        load_spp(path=path)

    else:
        cv = cross_validation(corpus=corpus, n_split=5)
        vocab_sizes = [50, 100, 500, 1000, 2000]

        result = {}
        for vocab_size in vocab_sizes:
            for split_index, (train_corpus, test_corpus) in enumerate(cv):
                print("=" * 40)
                print(f"===== Split number {split_index + 1} is test split =====")
                print("=" * 40)
                io_model = train_spp(corpus=train_corpus, vocab_size=vocab_size)
                sp = spm.SentencePieceProcessor(model_proto=io_model.getvalue())
                unk_rate = calculate_unk_rate(corpus=test_corpus, sp=sp)
                result[(split_index, vocab_size)] = unk_rate

                print(f"Vocab size: {vocab_size}\nUNK rate: {unk_rate}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--textual-source', action='store', type=str, required=False, default='cryptonews')
    parser.add_argument('--labeling-method', action='store', type=str, required=False, default='color')

    parser.add_argument('--load-spp', action='store_true', required=False, default=False)
    parser.add_argument('--save-spp', action='store_true', required=False, default=False)
    args = parser.parse_args()

    run_sentence_piece(args)


if __name__ == '__main__':
    main()
