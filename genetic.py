# this code is supposed to cover 1 section
# from udemy course about genetic algorithms
import os
import re
import math
import random
import string
from typing import Dict, List

import requests
import numpy as np

regex = re.compile("[^a-zA-Z]")


def get_alphabet_map():
    return dict(zip(string.ascii_lowercase, range(26)))


def get_coding_map() -> Dict[str, str]:
    # get coding map
    l1 = list(string.ascii_lowercase)
    l2 = list(string.ascii_lowercase)

    random.shuffle(l2)

    coding_map = dict(zip(l1, l2))
    coding_map[" "] = " "

    return coding_map


def encrypt(coding_map: Dict[str, str], message: str) -> str:
    msg = message.lower()
    msg = regex.sub(" ", msg)

    return "".join([coding_map[l] for l in msg])


def decrypt(coding_map: Dict[str, str], message: str) -> str:
    return "".join([coding_map[l] for l in message])


def load_book():
    if not os.path.exists("moby_dick.txt"):
        print("Downloading moby dick...")

        r = requests.get("https://lazyprogrammer.me/course_files/moby_dick.txt")

        with open("moby_dick.txt", "w") as f:
            f.write(r.content.decode())


def evolve_offspring(
    dna_pool: List[Dict[str, str]], child_count: int
) -> List[Dict[str, str]]:
    result_pool = []

    for dna in dna_pool:
        for _ in range(child_count):
            # swap values - mutate parent
            dna_copy = {k: v for k, v in dna.items()}
            keys = list(dna_copy.keys())

            tmp = " "
            while tmp == " ":
                i = random.randint(0, len(dna) - 1)
                tmp = dna_copy[keys[i]]

            tmp2 = " "
            while tmp2 == " ":
                j = random.randint(0, len(dna) - 1)
                tmp2 = dna_copy[keys[j]]

            dna_copy[keys[i]] = tmp2
            dna_copy[keys[j]] = tmp

            result_pool.append(dna_copy)

    return dna_pool + result_pool


class LangMachine:
    def __init__(self, text: str):
        self.alphabet_map = get_alphabet_map()

        self.M = np.ones((26, 26))
        self.pi = np.zeros(26)

        regex = re.compile("[^a-zA-Z]")

        for line in text.split("\n"):
            line = line.rstrip()

            if not line:
                continue

            line = regex.sub(" ", line)

            for word in line.lower().split():
                self._build_unigram(word[0])

                prev_char = word[0]
                for letter in word[1:]:
                    self._build_bigram(prev_char, letter)
                    prev_char = letter

        self.pi /= self.pi.sum()
        self.M /= self.M.sum(axis=1, keepdims=True)

    def get_word_prod(self, word: str) -> float:
        result_prob = 0.0
        first_letter = word[0]

        result_prob += math.log(self.pi[self.alphabet_map[first_letter]])

        prev_letter = first_letter

        for letter in word[1:]:
            prev_letter_pos = self.alphabet_map[prev_letter]
            letter_pos = self.alphabet_map[letter]
            result_prob += math.log(self.M[prev_letter_pos, letter_pos])

            prev_letter = letter

        return result_prob

    def get_word_sequence_prod(self, sequence: str):
        result_prob = 0.0

        for word in sequence.split():
            result_prob += self.get_word_prod(word)

        return result_prob

    def _build_unigram(self, letter: str):
        i = self.alphabet_map[letter]
        self.pi[i] += 1

    def _build_bigram(self, prev_char: str, current_char: str):
        i = self.alphabet_map[prev_char]
        j = self.alphabet_map[current_char]
        self.M[i, j] += 1


if __name__ == "__main__":
    load_book()

    with open("moby_dick.txt") as book:
        text = book.read().lower()

        lang_machine = LangMachine(text)

        original_coding_map = get_coding_map()

        message = """I then lounged down the street and found,
    as I expected, that there was a mews in a lane which runs down
    by one wall of the garden. I lent the ostlers a hand in rubbing
    down their horses, and received in exchange twopence, a glass of
    half-and-half, two fills of shag tobacco, and as much information
    as I could desire about Miss Adler, to say nothing of half a dozen
    other people in the neighbourhood in whom I was not in the least
    interested, but whose biographies I was compelled to listen to.
        """

        encrypted_text = encrypt(original_coding_map, message)
        print("encrypted_text", encrypted_text)

        num_offsping = 1000
        best_count = 5
        dna_pool_size = 20

        dna_pool = [get_coding_map() for _ in range(dna_pool_size)]

        for i in range(num_offsping):
            probs = []
            for dna in dna_pool:
                decrypted_text = decrypt(dna, encrypted_text)
                prob = lang_machine.get_word_sequence_prod(decrypted_text)
                probs.append((prob, dna))

            sorted_probs = sorted(probs, key=lambda prob: prob[0], reverse=True)

            best = [b[1] for b in sorted_probs[:best_count]]
            dna_pool = evolve_offspring(
                best, child_count=(dna_pool_size // best_count) - 1
            )

        for k, v in original_coding_map.items():
            if best[0][k] == v:
                print("correct guess", k, v)

        decrypted_text = decrypt(best[0], encrypted_text)
        print(decrypted_text)

        original_msg_prob = lang_machine.get_word_sequence_prod(
            regex.sub(" ", message.lower())
        )
        decrypted_msg_prob = lang_machine.get_word_sequence_prod(decrypted_text)

        print("original message prob", original_msg_prob)
        print("decrypted message prob", decrypted_msg_prob)
