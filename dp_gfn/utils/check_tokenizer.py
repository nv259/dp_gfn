import os
import sys

from transformers import AutoTokenizer

from dp_gfn.utils.data import BaseDataset


class TestTokenizerDataset(BaseDataset):
    def __call__(self, tokenizer="FacebookAI/xlm-roberta-base"):
        check = True

        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        for item in self.data:
            tokens = tokenizer(item["text"], add_special_tokens=False)
            tok_num_words = get_num_words(tokens)

            if tok_num_words != item["num_words"]:
                print("-" * 20)
                print(item["text"])
                print(tokens.tokens())
                print(tok_num_words, item["num_words"])
                print("-" * 20)
                print()
                check = False

        return check


def get_num_words(tokens):
    num_words = 0
    start, end = 0, 0

    for word_id in tokens.word_ids():
        if word_id is None:
            continue

        start, end = tokens.word_to_tokens(word_id)
        num_words = max(num_words, word_id)

    return num_words


# print(sys.argv[1], sys.argv[2])
if __name__ == "__main__":
    parent_dataset_folder = sys.argv[1]
    check = True

    for dataset_folder in os.listdir(parent_dataset_folder):
        if sys.argv[2].lower() in dataset_folder.lower():
            dataset_folder = os.path.join(parent_dataset_folder, dataset_folder)
            for file in os.listdir(dataset_folder):
                if file.endswith(".conllu"):
                    print(file)
                    path_to_file = os.path.join(dataset_folder, file)

                    try:
                        check = TestTokenizerDataset(path_to_file)(sys.argv[3])
                    except:
                        check = TestTokenizerDataset(path_to_file)()

    if check:
        print("Tokenizer is applicable for all datasets.")
    else:
        print("Tokenizer is not applicable for", sys.argv[2], "datasets.")
