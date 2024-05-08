#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniil Kocharov (dan_ya)
@description: Parse AOCHILDES csv files and create .txt files for LM training through TensorFlow datasets.
"""
from argparse import ArgumentParser
from collections import defaultdict
from csv import DictReader
from pathlib import Path
import os
import re
import shutil

children = {'SIS', 'BRO', 'SIB', 'BOY', 'GIRL', 'FND'}
target_child = {'CHI'}
mother = {'MOT'}
father = {'FAT'}


def parse_args(parser):
    parser.add_argument("--age-interval", type=int, default=3, help="Months in age interval to be saved as one bin.")
    parser.add_argument("--age-limit", type=int, default=90, help="Age limit in months.")
    parser.add_argument("--clear-datasets", action="store_true", default=False, help="Remove previously created output files.")
    parser.add_argument("--min-utterance-length", type=int, default=1, help="Minimum number of words in an utterance.")
    parser.add_argument("--input-dir", type=str, default=os.path.join('original_transcripts'), help="Input directory path.")
    parser.add_argument("--output-dir", type=str, default=os.path.join('ao_dataset'), help="Output directory path.")
    return parser.parse_args()


def main():
    parser = ArgumentParser(description='Parse AOCHILDES csv files and create .txt files for LM training',
                            allow_abbrev=False)
    args = parse_args(parser)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    assert input_dir.is_dir(), f"No input directory: {input_dir}"
    
    if args.clear_datasets:
        shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    csv_files = sorted(input_dir.glob('*.csv'))
    dataset = defaultdict(lambda: defaultdict(list))  # dataset['age']['speaker_id'] = list(text)
    for ifile, file in enumerate(csv_files):
        print(f'Processing corpus {file.stem} ({ifile+1} out of {len(csv_files)}).')
        data = read_csv_file(file)

        for unit in data:
            speaker = unit['speaker_code']
            if speaker not in mother and speaker not in father:
                continue
            if unit['gloss'] == '':
                continue
            if '0' in unit['gloss'] or 'xxx' in unit['gloss'] or 'yyy' in unit['gloss'] or 'www' in unit['gloss']:
                continue

            text = re.sub(r'\s+', ' ', unit['gloss'].strip())
            text = normalize_text(text)
            words = text.split(' ')
            words = remove_non_english_words(words, unit['id'])
            if len(words) < args.min_utterance_length:
                continue
            words += ['.']
            text = ' '.join(words)

            age = unit['target_child_age']
            if age in {'', 'NA'}:
                age = 'unk'
            else:
                age = float(age)
                age = int((age+args.age_interval/2)/args.age_interval) * args.age_interval
                if age > args.age_limit:
                    age = 'unk'
                else:
                    age = 'age_' + str(age).zfill(3)
            speaker_id = f"{file.stem}_{speaker}"
            dataset[age][speaker_id].append(text)

    for age in sorted(dataset):
        dir_name = Path(output_dir, f"childes_{age}")
        os.makedirs(dir_name, exist_ok=True)
        age_bin_size = 0
        for speaker_id in dataset[age]:
            age_bin_size += len(dataset[age][speaker_id])
            text_file = Path(dir_name, f'{speaker_id}.txt')
            with open(text_file, 'w', encoding='utf-8') as fo:
                text = '\n'.join(dataset[age][speaker_id])
                fo.write(text)
        print(f"childes_{age}: {age_bin_size}")
    return True


def read_csv_file(csv_file):
    data = []
    with open(csv_file, newline='', encoding='utf-8') as csvfile:
        reader = DictReader(csvfile)
        for row in reader:
            data.append(row)
    return data


def normalize_text(text_string: str):
    """
    Removes odd symbols, as + and _ from word tokens.
    """
    text_string = re.sub(r'(\b\w+)\+\1', r'\1', text_string)
    text_string = text_string.replace('+', ' ')
    text_string = text_string.replace('_', ' ')
    text_string = text_string.replace(':', '')
    return text_string


def remove_non_english_words(tokens, text_id):
    good_pattern = r"['a-zA-Z0-9 -]+$"
    good_tokens = [t for t in tokens if re.match(good_pattern, t)]
    bad_tokens = [(text_id, t) for i, t in enumerate(tokens) if not re.match(good_pattern, t)]
    if len(bad_tokens):
        print(bad_tokens)
    return good_tokens


if __name__ == '__main__':
    main()
