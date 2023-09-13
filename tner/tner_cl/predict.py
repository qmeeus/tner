""" command line tool to test finetuned NER model """
import logging
import argparse
import json
from pathlib import Path
from pprint import pprint
from functools import partial

from transformers import pipeline
from tner import TransformersNER
from tner.util import tokenize_sentence

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def get_options():
    parser = argparse.ArgumentParser(description='command line tool to test finetuned NER model',)
    parser.add_argument('-m', '--model', help='model alias of huggingface or local checkpoint', required=True, type=str)
    parser.add_argument('-f', '--file', help='test file with "<id> <sentence>" per line', default=None, type=str)
    parser.add_argument('-o', '--output', help='output file in json lines format', default=None, type=str)
    parser.add_argument('-t', '--translate', help='translasion model for translating input sentence', default=None, type=str)
    parser.add_argument('-b', '--batch_size', help='batch size', default=32, type=int)
    parser.add_argument('-n', '--max_samples', help='predict only n samples', type=int, default=None)
    return parser.parse_args()


def read_sentences(filename, max_samples=None):
    i = 0
    with open(filename, 'r') as f:
        for line in f:
            try:
                id, sentence = line.strip().split(maxsplit=1)
                yield id, sentence
                i += 1
            except ValueError:
                logging.warning(f'No text found: {line}')
            if max_samples and i == max_samples:
                break

def main():
    opt = get_options()

    translate = None
    if opt.translate:
        translate = pipeline('translation', model=opt.translate)

    classifier = TransformersNER(opt.model)

    if opt.file:
        if opt.output is None:
            opt.output = Path(opt.file).stem + '.jsonl'
        print(f'writing results to {opt.output}')
        with open(opt.output, 'w') as output_file:
            ids, sentences = map(list, zip(*read_sentences(opt.file, max_samples=opt.max_samples)))
            if translate:
                sentences = [x["translation_text"] for x in translate(sentences)]
            keys = ["input", "entity_prediction", "prediction", "probability", "nll"]
            tokenized_sentences = list(map(tokenize_sentence, sentences))
            results = classifier.predict(tokenized_sentences, batch_size=opt.batch_size)
            predictions = [ids] + [results[key] for key in keys]
            keys = ["id"] + keys
            for pred in zip(*predictions):
                print(json.dumps(dict(zip(keys, pred)), ensure_ascii=False), file=output_file)
        return

    sentences = [
        'I live in United States.',
        'I have an Apple computer.',
        'I like to eat an apple.'
    ]
    test_result = classifier.predict(sentences)
    pprint('-- DEMO --')
    pprint(test_result)
    pprint('----------')
    while True:
        _input = input('input sentence >>> ')
        if _input == 'q':
            break
        elif _input == '':
            continue
        else:
            pprint(classifier.predict([_input]))

