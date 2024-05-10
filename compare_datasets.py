#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniil Kocharov (dan_ya)
"""
from argparse import ArgumentParser
import copy
import pickle
from collections import defaultdict
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
import re
from scipy.stats import spearmanr
import stanza
from stanza.models.common.doc import Document, Sentence
import time
from typing import Callable, Union, Any
import analysis_functions

nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')
pos_to_analyze = ('ADJ', 'ADV', 'INTJ', 'NOUN', 'PROPN', 'VERB',                 # Open class words
                  'ADP', 'AUX', 'CCONJ', 'DET', 'NUM', 'PART', 'PRON', 'SCONJ')  # Close class words


def parse_args(parser):
    # Pipeline flags
    parser.add_argument('--sample_data', action='store_true', default=False, help='Sample the data.')
    parser.add_argument('--analyse_data', action='store_true', default=False, help='Analyse the data samples.')
    parser.add_argument('--update_features', action='store_true', default=False, help='Update feature measures.')
    parser.add_argument('--features_to_update', default='lm_perplexity', help='Name of the analytical measure to be recalculated')
    parser.add_argument('--statistical_tests', action='store_true', default=False, help='Perform statistical tests.')
    parser.add_argument('--plot_figures', action='store_true', default=False, help='Plot figures.')
    parser.add_argument('--show_figures', action='store_true', default=False, help='Show figures before saving.')
    parser.add_argument('--violinplot_with_quartiles', action='store_true', default=True, help='Draw quartiles in violin plots.')

    # Paths to directories and files
    parser.add_argument("--input-path", type=str, default=os.path.join('ao_dataset'), help="Input directory or file path.")
    parser.add_argument("--output-path", type=str, default=os.path.join('output'), help="Output directory path.")
    parser.add_argument('--analysis_parameter_file', default='analysis_parameters.json', help='JSON with analysis parameters')

    # Other parameters
    parser.add_argument('--reference_dataset', default='childes_age_057', help='Reference dataset name')
    parser.add_argument('--analytics_required', default='', help='The name of the analytical measure to be tested')
    parser.add_argument('--token_count_limit', type=int, default=0, help='Minimal number of tokens for a word to be counted')
    parser.add_argument('--sentence_max_length', type=int, default=0, help='The maximum length of a sentence (in words). Longer sentences will not be loaded. 0 - for no limitation.')
    parser.add_argument('--dataset_min_size', type=int, default=2000, help='The minimum size of a dataset (in sentences). Smaller datasets will not be loaded. 0 - for no limitation.')
    parser.add_argument('--sampling_size', type=int, default=100, help='Size of sampling distribution to be compared')
    parser.add_argument('--sampling_tokens', type=int, default=1000, help='Sampling: number of tokens for token-wise comparison')
    parser.add_argument('--sampling_sentences', type=int, default=100, help='Sampling: number of sentences for sentence-wise comparison')
    parser.add_argument('--sampling_batches_n', type=int, default=100, help='Sampling: number of batches for perplexity calculation')
    parser.add_argument('--sampling_batches_size', type=int, default=50, help='Sampling: size of batches for perplexity calculation')
    return parser.parse_args()


parser = ArgumentParser(description='Parse AOCHILDES csv files and create .txt files for LM training', allow_abbrev=False)
args = parse_args(parser)


def main():

    input_path = Path(args.input_path)
    output_path =Path(args.output_path)
    report_dir = Path(output_path, f'reports')
    figure_dir = Path(output_path, f'img')
    sampling_dir = Path(output_path, f'samples')
    feature_file = Path(report_dir, f'features.txt')
    statistical_report_file = Path(report_dir, f'statistical_report.txt')
    for d in [report_dir, sampling_dir, figure_dir]:
        os.makedirs(d, exist_ok=True)

    with open(args.analysis_parameter_file, 'r') as fi:
        data = '\n'.join(line for line in fi if not line.strip().startswith('#'))
        analysis_setup = json.loads(data)

    if args.analytics_required == '':
        analytics_required = sorted(analysis_setup)
        analytics_required = [f for f in analytics_required if analysis_setup[f]['function'] is not None]
    else:
        analytics_required = [args.analytics_required]

    if args.sample_data:
        print('\nsampling dataset...')
        datasets = load_datasets(input_path)
        dataset_samples = dataset_sampling(datasets)
        save_dataset_samples(dataset_samples, sampling_dir)

    if args.analyse_data:
        print('\nfeature extraction...')
        print('- loading sampled dataset...')
        dataset_names = get_dataset_names(input_path)
        dataset_samples = load_sampled_datasets(sampling_dir, dataset_names)
        if dataset_samples is None:
            print('- sampling dataset...')
            datasets = load_datasets(input_path)
            dataset_samples = dataset_sampling(datasets)
        feature_distributions = defaultdict(dict)
        for ds in sorted(dataset_samples):
            print(f'\n{ds}')
            for analysis in analytics_required:
                sample_type = analysis_setup[analysis]['sampling']
                print('---', analysis)
                if sample_type is None:
                    feature_distributions[ds][analysis] = [dataset_samples[ds][analysis] for i in range(args.sampling_size)]
                elif analysis == 'pos_rate':
                    feature_measures = analyze(getattr(analysis_functions, analysis_setup[analysis]['function']), dataset_samples[ds][sample_type], pos=pos_to_analyze)
                    for f in feature_measures:
                        print('------', f)
                        f_label = f"{analysis}_{f}"
                        feature_distributions[ds][f_label] = feature_measures[f]
                elif analysis == 'voc_rank_diff':
                    reference_dataset = get_reference_dataset(args.reference_dataset, sampling_dir)
                    reference_dataset = list(reference_dataset.values())[0]
                    reference_dataset = reference_dataset[sample_type]
                    feature_distributions[ds][analysis] = analyze(getattr(analysis_functions, analysis_setup[analysis]['function']), (dataset_samples[ds][sample_type], reference_dataset))
                else:
                    feature_distributions[ds][analysis] = analyze(getattr(analysis_functions, analysis_setup[analysis]['function']), dataset_samples[ds][sample_type])
            df = pd.DataFrame(feature_distributions[ds])
            ds_report_path = os.path.join(report_dir, f'{ds}.features')
            df.to_csv(ds_report_path, sep='\t', index=False)
        report = json.dumps(feature_distributions, ensure_ascii=False, indent=4, sort_keys=True)
        with open(feature_file, 'w', encoding='utf-8') as fo:
            fo.write(report)
    else:
        with open(feature_file, 'r', encoding='utf-8') as fi:
            feature_distributions = json.load(fi)

    # update features if necessary
    if args.update_features:
        print('\nupdate feature information...')
        some_ds_name = list(feature_distributions.keys())[0]
        measured_features = list(feature_distributions[some_ds_name].keys())
        analytics_to_be_added = [f for f in analytics_required if f not in measured_features]
        analytics_to_be_added = [f for f in analytics_to_be_added if len([m for m in measured_features if m.startswith(f)]) == 0]
        analytics_to_be_added += [args.features_to_update]
        if len(analytics_to_be_added) != 0:
            dataset_names = get_dataset_names(input_path)
            dataset_samples = load_sampled_datasets(sampling_dir, dataset_names)
            if dataset_samples is None:
                datasets = load_datasets(input_path)
                dataset_samples = dataset_sampling(datasets)
            for ds in sorted(dataset_samples):
                print(f'\n{ds}')
                for analysis in analytics_to_be_added:
                    sample_type = analysis_setup[analysis]['sampling']
                    print('---', analysis)
                    if sample_type is None:
                        feature_distributions[ds][analysis] = [dataset_samples[ds][analysis] for i in range(args.sampling_size)]
                    elif analysis == 'pos_rate':
                        feature_measures = analyze(getattr(analysis_functions, analysis_setup[analysis]['function']), dataset_samples[ds][sample_type], pos=pos_to_analyze)
                        for f in feature_measures:
                            print('------', f)
                            f_label = f"{analysis}_{f}"
                            feature_distributions[ds][f_label] = feature_measures[f]
                    elif analysis == 'voc_rank_diff':
                        reference_dataset = get_reference_dataset(args.reference_dataset, sampling_dir)
                        reference_dataset = list(reference_dataset.values())[0]
                        reference_dataset = reference_dataset[sample_type]
                        feature_distributions[ds][analysis] = analyze(getattr(analysis_functions, analysis_setup[analysis]['function']), (dataset_samples[ds][sample_type], reference_dataset))
                    else:
                        feature_distributions[ds][analysis] = analyze(getattr(analysis_functions, analysis_setup[analysis]['function']), dataset_samples[ds][sample_type])
                df = pd.DataFrame(feature_distributions[ds])
                ds_report_path = os.path.join(report_dir, f'{ds}.features')
                df.to_csv(ds_report_path, sep='\t', index=False)
            report = json.dumps(feature_distributions, ensure_ascii=False, indent=4, sort_keys=True)
            with open(feature_file, 'w', encoding='utf-8') as fo:
                fo.write(report)

    # Transpose data in terms of 'feature', 'dataset' factors
    transposed_data = defaultdict(dict)
    for ds in feature_distributions:
        for f in feature_distributions[ds]:
            transposed_data[f][ds] = feature_distributions[ds][f]
    feature_distributions = transposed_data

    if args.plot_figures:
        print(f'\nplotting figures...')
        for analysis in sorted(feature_distributions):
            print(f'--- {analysis}')
            if analysis.startswith('pos_rate'):
                pos = analysis.split('_')[-1]
                plot_title = f"{analysis_setup['pos_rate']['plot_title']} - {pos}"
                plotting_style = analysis_setup['pos_rate']['plot_style']
            else:
                plot_title = analysis_setup[analysis]['plot_title']
                plotting_style = analysis_setup[analysis]['plot_style']

            if plotting_style == 'line':
                data_to_plot = {k: np.mean(feature_distributions[analysis][k]) for k in feature_distributions[analysis]}
            else:
                data_to_plot = feature_distributions[analysis]

            if plotting_style == 'line':
                plot_analytics_with_lines(data_to_plot, plot_title, figure_dir)
            elif plotting_style == 'boxplot':
                plot_analytics_with_boxplots(data_to_plot, plot_title, figure_dir)
            elif plotting_style == 'violinplot':
                plot_analytics_with_violinplots(data_to_plot, plot_title, figure_dir)

    if args.statistical_tests and 'n_words' in feature_distributions:
        print(f'\nstatistical tests...')
        ref_metrics = ['age', 'n_words']
        features = sorted([m for m in list(feature_distributions.keys()) if m not in ref_metrics])
        if 'age' in ref_metrics and 'age' not in feature_distributions:
            non_relevant_datasets = [ds for ds in feature_distributions[features[0]] if 'age' not in ds]
            for metric in feature_distributions:
                for ds in non_relevant_datasets:
                    del feature_distributions[metric][ds]
            feature_distributions['age'] = dict()
            for ds in feature_distributions[features[0]]:
                age_id = ds.split('_')[-1]
                feature_distributions['age'][ds] = [int(age_id) for i in range(args.sampling_size)]

        statistical_report = []
        for rm in ref_metrics:
            r_values = [np.mean(feature_distributions[rm][ds]) for ds in sorted(feature_distributions[rm])]
            for mm in features:
                m_values = [np.mean(feature_distributions[mm][ds]) for ds in sorted(feature_distributions[rm])]
                r, p = spearmanr(r_values, m_values)
                unit = [rm, mm, round(r, 2), round(p, 3)]
                statistical_report.append(unit)
        # check if reference measures change monotonously.
        for rm in ref_metrics:
            r_values = [np.mean(feature_distributions[rm][ds]) for ds in sorted(feature_distributions[rm])]
            r, p = spearmanr(r_values, list(range(len(r_values))))
            unit = ['-', rm, round(r, 2), round(p, 3)]
            statistical_report.append(unit)

        with open(statistical_report_file, 'w', encoding='utf-8') as fo:
            for line in statistical_report:
                line = '\t'.join([str(v) for v in line]) + '\n'
                fo.write(line)

    return True


def get_reference_dataset(ds_name: str, sampling_dir: Path):
    ds_id = ds_name.split(os.sep)
    if ds_id[-1] != '':
        ds_id = ds_id[-1]
    else:
        ds_id = ds_id[-2]
    ds_id, _ = os.path.splitext(ds_id)
    potential_sampling_path = Path(sampling_dir, f"{ds_id}.pkl")
    if potential_sampling_path.is_file():
        return load_sampled_datasets(sampling_dir, [ds_id])
    datasets = load_datasets(Path(ds_name))
    dataset_samples = dataset_sampling(datasets)
    save_dataset_samples(dataset_samples, sampling_dir)
    return dataset_samples


def analyze(function_name: Callable, dataset: Union[list[str], tuple[Any, Any]], **kwargs):
    if len(kwargs) == 0:
        metric = function_name(copy.deepcopy(dataset))
    else:
        metric = function_name(copy.deepcopy(dataset), arg=kwargs)
    return metric


def dataset_sampling(datasets):
    sampling_datasets = {}
    for d in sorted(datasets):
        print(d, len(datasets[d]['txt']), len(datasets[d]['parsed']))
        sampling_datasets[d] = {'tokens': [], 'sentences': [], 'batches': []}
        dataset_tokens = [w for sentence in datasets[d]['parsed'] for w in sentence.words]
        dataset_tokens = [w for w in dataset_tokens if w.pos != 'PUNCT']
        dataset_batches = _split_into_batches(datasets[d]['parsed'], args.sampling_batches_size)
        for i in range(args.sampling_size):
            n = min(args.sampling_tokens, len(dataset_tokens))
            samples = np.random.choice(dataset_tokens, size=n, replace=False)
            sampling_datasets[d]['tokens'].append(samples)

            n = min(args.sampling_sentences, len(datasets[d]['parsed']))
            samples = np.random.choice(datasets[d]['parsed'], size=n, replace=False)
            sampling_datasets[d]['sentences'].append(samples)

            n = min(args.sampling_batches_n, len(dataset_batches))
            samples = np.random.choice(dataset_batches, size=n, replace=False)
            sampling_datasets[d]['batches'].append(samples)
        sampling_datasets[d]['n_words'] = len(dataset_tokens)
        sampling_datasets[d]['voc_size'] = len(set([w.lemma for w in dataset_tokens]))
    return sampling_datasets


def get_dataset_names(dataset_path: Path) -> list[str]:
    datasets = list()
    if dataset_path.is_file() and dataset_path.suffix == '.txt':
        datasets.append(dataset_path.stem)
    elif dataset_path.is_dir():
        ds_files = sorted(dataset_path.glob('*.txt'))
        ds_dirs = sorted(dataset_path.glob('**/'))
        datasets += [d.stem for d in ds_dirs]
        datasets += [f.stem for f in ds_files]
    return datasets


def load_datasets(dataset_path: Path) -> dict[str, dict[str, list[str]]]:
    """
    Loads datasets. The path could be a path to:
     - single dataset as a path to .txt file,
     - directory with multiple datasets as TXT-files and subdirectories (each of subdirectories contains .txt files).

    Parameters:
    dataset_paths (Path): path to datasets.
    args (Namespace): global arguments.

    Returns:
    dictionary of datasets, as following: {'dataset1_id': dataset1, .., 'datasetN_id': datasetN},
    where each dataset is a dictionary of the following format:
    {'txt': <list of sentences>, 'parsed': <list of parsed sentences>}.
    """
    ds_min_size = max(1, args.dataset_min_size)
    datasets = defaultdict(dict)
    if dataset_path.is_file() and dataset_path.suffix == '.txt':
        print(dataset_path)
        datasets[dataset_path.stem]['txt'] = read_dataset_from_file(dataset_path)
        path_to_parsed = dataset_path.with_suffix('.parsed')
        datasets[dataset_path.stem]['parsed'] = parse_dataset(datasets[dataset_path.stem]['txt'], path_to_parsed, dataset_path.stem)
    elif dataset_path.is_dir():
        ds_files = sorted(dataset_path.glob('*.txt'))
        ds_dirs = sorted(dataset_path.glob('**/'))
        for ds_file in ds_files:
            print(ds_file)
            datasets[ds_file.stem]['txt'] = read_dataset_from_file(ds_file)
            ds_parsed = ds_file.with_suffix('.parsed')
            datasets[ds_file.stem]['parsed'] = parse_dataset(datasets[dataset_path.stem]['txt'], ds_parsed, ds_file.stem)
        for ds_dir in ds_dirs:
            print(ds_dir.stem)
            datasets[ds_dir.stem]['txt'] = read_dataset_from_dir(ds_dir)
            ds_parsed = ds_dir.with_suffix('.parsed')
            datasets[ds_dir.stem]['parsed'] = parse_dataset(datasets[ds_dir.stem]['txt'], ds_parsed, ds_dir.stem)
    for d in datasets:
        datasets[d] = _remove_long_sentences(datasets[d], args.sentence_max_length)
    empty_dataset_ids = [d for d in datasets if len(datasets[d]['txt']) < ds_min_size]
    for d in empty_dataset_ids:
        del datasets[d]
    if 'childes_unk' in datasets:
        del datasets['childes_unk']
    assert len(datasets) != 0, 'No valid datasets to analyze!'

    print('\ndatasets loaded')
    for d in datasets:
        print(d, len(datasets[d]['txt']))
    return datasets


def load_sampled_datasets(input_dir: Path, list_of_dataset=None):
    files = sorted(input_dir.glob('*.pkl'))
    if len(files) == 0:
        return None
    dataset = {}
    for file in sorted(files):
        if list_of_dataset is not None and file.stem not in list_of_dataset:
            continue
        with open(file, 'rb') as fi:
            dataset[file.stem] = pickle.load(fi)
    return dataset


def parse_dataset(text: Union[str, list[str]], filepath: Path, ds_name: str) -> Union[list[Sentence], None]:
    if filepath.is_file():
        with open(filepath, 'rb') as fo:
            parsed_text = Document.from_serialized(fo.read())
        if len(parsed_text.sentences) != 0:
            parsed_text = [s for s in parsed_text.sentences]
            return parsed_text
    print(ds_name, '\n---', 'parsing')
    if type(text) is list:
        text = ' '.join(text)
    if type(text) is not str:
        return None
    timestamp = time.time()
    parsed_text = nlp(text)
    minutes, seconds = divmod(time.time()-timestamp, 60)
    print('---', f"parsing of {text.count('.')} sentences: {round(minutes)}:{round(seconds, 2)}.")
    data = parsed_text.to_serialized()
    with open(filepath, 'wb') as fo:
        fo.write(data)
    parsed_text = [s for s in parsed_text.sentences]
    return parsed_text


def plot_analytics_with_violinplots(data: dict[str, Union[list[float], list[int]]], label: str, figure_path: Path):
    labels = sorted(data)
    if args.violinplot_with_quartiles:
        data = copy.deepcopy(data)
        for k in labels:
            cutoff = 2.5
            lower_limit = np.percentile(data[k], cutoff)
            upper_limit = np.percentile(data[k], 100-cutoff)
            data[k] = [v for v in data[k] if lower_limit <= v <= upper_limit]
    plt.violinplot([data[k] for k in labels])
    plt.title(f'{label}', fontweight='bold')
    x_ticks = [x+1 for x in range(len(labels))]
    x_labels = [_clear_x_tick_label(k) for k in labels]
    plt.xticks(x_ticks, x_labels, rotation='vertical')
    plt.grid(True)
    if args.show_figures:
        plt.show()
    figure_file = Path(figure_path, f'{label}.png')
    plt.savefig(figure_file, bbox_inches='tight')
    plt.savefig(figure_file.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()


def plot_analytics_with_boxplots(data: dict[str, Union[float, int]], label: str, figure_path: Path):
    labels = sorted(data)
    plt.boxplot([data[k] for k in labels], labels=[_clear_x_tick_label(k) for k in labels], showfliers=False)
    plt.title(f'{label}', fontweight='bold')
    plt.xticks(rotation=90)
    plt.xlabel('age')
    plt.grid(True)
    if args.show_figures:
        plt.show()
    figure_file = Path(figure_path, f'{label}.png')
    plt.savefig(figure_file, bbox_inches='tight')
    plt.savefig(figure_file.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()


def plot_analytics_with_lines(data: dict[str, Union[float, int]], label: str, figure_path: Path):
    labels = sorted(data)
    values = [data[k] for k in labels]
    x_ticks = list(range(len(labels)))
    x_labels = [_clear_x_tick_label(k) for k in labels]
    plt.plot(values)
    plt.title(f'{label}', fontweight='bold')
    plt.xticks(x_ticks, x_labels, rotation='vertical')
    plt.xlabel('age')
    if args.show_figures:
        plt.show()
    figure_file = Path(figure_path, f'{label}.png')
    plt.savefig(figure_file, bbox_inches='tight')
    plt.savefig(figure_file.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()


def read_dataset_from_dir(path: Path) -> list[str]:
    if not path.is_dir():
        print(f"No directory: {path}")
        return []
    files = sorted(path.glob('*.txt'))
    if len(files) == 0:
        return []
    text = []
    for file in files:
        text += read_dataset_from_file(file)
    return text


def read_dataset_from_file(path: Path) -> list[str]:
    if not path.is_file():
        print(f"No file: {path}")
        return []
    with open(path, 'r', encoding='utf-8') as fi:
        text = fi.readlines()
    text = [line.strip() for line in text]
    text = [line for line in text if line != '']
    text = [line.split('\t')[-1] for line in text if not line.startswith('#')]
    text = ' '.join(text)
    text = re.sub(r'(?<=[.?!]) ', '$$$', text)
    text = text.split('$$$')
    text = [t.strip() for t in text]
    text = [t for t in text if t != '']
    return text


def save_dataset_samples(data, output_dir):
    for ds_id in data:
        bin_path = Path(output_dir, f"{ds_id}.pkl")
        with open(bin_path, 'wb') as fo:
            pickle.dump(data[ds_id], fo)
    return None


def _clear_x_tick_label(text: str) -> str:
    text = os.path.split(text)[-1].replace('.samples', '').replace('gpt.model.', '').replace('childes_age_0', '')
    return text


def _remove_long_sentences(dataset, max_sentence_length: int) -> Union[list[str], list[Sentence]]:
    if max_sentence_length > 0:
        bad_ids = [i for i in range(len(dataset['parsed'])) if len(dataset['parsed'][i].words) < max_sentence_length]
        for i in reversed(bad_ids):
            del dataset['parsed'][i]
            del dataset['txt'][i]
    return dataset


def _split_into_batches(text, batch_limit):
    batches = []
    batch = []
    batch_length = 0
    for sentence in text:
        if batch_limit < batch_length + len(sentence.words):
            batch.append(sentence)
            batches.append(' '.join([s.text for s in batch]))
            batch = []
            batch_length = 0
        else:
            batch.append(sentence)
            batch_length += len(sentence.words)
    return batches


if __name__ == '__main__':
    main()
