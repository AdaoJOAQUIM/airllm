import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from airllm.induction import (CTWPredictor, mdl_program_search, ncd,
                              predict_by_compression)


class TestCTW(unittest.TestCase):

    def bits_per_symbol(self, bits, depth=8):
        p = CTWPredictor(depth=depth)
        p.update_bits(bits)
        return p.logloss_bits() / len(bits)

    def test_structured_sequences_compress(self):
        # a good universal predictor must drive the code length of
        # structured data far below 1 bit/symbol
        self.assertLess(self.bits_per_symbol([0] * 512), 0.1)
        self.assertLess(self.bits_per_symbol([i % 2 for i in range(512)]), 0.1)

    def test_random_sequence_hits_shannon_floor(self):
        # ... and CANNOT beat 1 bit/symbol on true randomness (Shannon)
        rng = random.Random(0)
        bits = [rng.randint(0, 1) for _ in range(512)]
        bps = self.bits_per_symbol(bits)
        self.assertGreater(bps, 0.95)
        self.assertLess(bps, 1.1)

    def test_periodic_prediction_accuracy(self):
        pattern = [0, 1, 1]
        p = CTWPredictor(depth=6)
        for i in range(300):
            p.update(pattern[i % 3])
        correct = 0
        for i in range(300, 360):
            predicted = 1 if p.predict_one() > 0.5 else 0
            correct += (predicted == pattern[i % 3])
            p.update(pattern[i % 3])
        self.assertEqual(correct, 60)

    def test_probabilities_are_normalized(self):
        p = CTWPredictor(depth=4)
        p.update_bytes(b'information')
        prob_one = p.predict_one()
        self.assertGreater(prob_one, 0.0)
        self.assertLess(prob_one, 1.0)


class TestMDLProgramSearch(unittest.TestCase):

    def test_finds_odd_numbers(self):
        result = mdl_program_search([1, 3, 5, 7, 9])
        self.assertIsNotNone(result)
        self.assertEqual(result['prediction'], 11)

    def test_finds_squares(self):
        result = mdl_program_search([0, 1, 4, 9, 16])
        self.assertEqual(result['expr'], '(n * n)')
        self.assertEqual(result['size'], 3)
        self.assertEqual(result['prediction'], 25)

    def test_occam_prefers_the_shortest_program(self):
        # identity fits [0, 1, 2] and is a single atom -- nothing shorter
        result = mdl_program_search([0, 1, 2])
        self.assertEqual(result['expr'], 'n')
        self.assertEqual(result['size'], 1)

    def test_builds_constants_beyond_the_atoms(self):
        result = mdl_program_search([7, 7, 7])
        self.assertEqual(result['prediction'], 7)

    def test_returns_none_when_no_short_program_exists(self):
        self.assertIsNone(mdl_program_search([0, 1, 0, 2, 0, 3], max_size=9))
        self.assertIsNone(mdl_program_search([]))


class TestCompressionDistance(unittest.TestCase):

    def test_ncd_separates_related_from_random(self):
        e1 = (b"information theory studies the quantification storage "
              b"and communication of information") * 3
        e2 = (b"the theory of information concerns compression transmission "
              b"and storage of messages") * 3
        rng = random.Random(1)
        noise = bytes(rng.randrange(256) for _ in range(len(e1)))
        self.assertLess(ncd(e1, e2), ncd(e1, noise))

    def test_ncd_self_distance_is_small(self):
        text = b"the quick brown fox jumps over the lazy dog" * 5
        self.assertLess(ncd(text, text), 0.2)

    def test_predict_by_compression_continues_the_period(self):
        history = b'abcabcabc' * 30 + b'ab'
        self.assertEqual(predict_by_compression(history), ord('c'))


if __name__ == '__main__':
    unittest.main()
