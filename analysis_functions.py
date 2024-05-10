#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author: Daniil Kocharov (dan_ya)
"""
from collections import Counter, defaultdict
import evaluate  # huggingface module
import numpy as np
import stanza
from stanza.models.common.doc import Sentence, Word
from scipy.spatial.distance import jensenshannon
import torch
from typing import Union, Tuple


nlp = stanza.Pipeline(lang='en', processors='tokenize, mwt, pos, lemma, depparse')
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'


def calc_complex_sentence_rate(data: list[list[Sentence]]) -> Union[list[float], list[np.ndarray]]:
    """
    Calculates rate of sentences with more than one clause,
    i.e. having one of following dependency relations: csubj, ccomp, xcomp, advcl, acl.
    :param data: list sample texts. Each text is a list of sentences to be processed.
    """
    clause_deprel = {'csubj', 'ccomp', 'xcomp', 'advcl', 'acl'}
    feature_measures = []
    for text in data:
        text_values = []
        for s in text:
            value = min(1, len([w for w in s.words if w.deprel in clause_deprel]))
            text_values.append(value)
        feature_measures.append(np.mean(text_values))
    return feature_measures


def calc_lm_perplexity(data: list[str], **kwargs) -> Union[list[float], None]:
    """
    Calculates text perplexity given a certain model (default: 'gpt2').
    :param data: list sample texts to be evaluated. Each sample text is a list of strings.
    :param model_nlp: HuggingFace model to be used for evaluation (default if 'gpt2').
    """
    kwargs = kwargs.get('arg', None)  # get real **kwargs passed into function as 'arg' argument
    model_nlp = 'gpt2'  # default HF model to be used.
    if kwargs is not None:
        model_nlp = kwargs.get('model_nlp', model_nlp)
    perplexity = evaluate.load("perplexity", module_type="metric")
    feature_measures = []
    for text in data:
        try:
            result = perplexity.compute(predictions=text, model_id=model_nlp, device=device)
            feature_measures.append(result['mean_perplexity'])
        except Exception as e:
            print(e)
            return None
    return feature_measures


def calc_interjection_sentence_rate(data: list[list[Sentence]]) -> list[float]:
    """
    Calculates a rate of sentences with interjection (INTJ) as a root.
    :param data: list sample texts to be evaluated. Each sample text is a list of Sentence.
    """
    feature_measures = [_calc_rate_of_sentence_by_root(text, {'INTJ'}) for text in data]
    return feature_measures


def calc_nominal_sentence_rate(data: list[list[Sentence]]) -> list[float]:
    """
    Calculates a rate of nominal sentences, defined as sentences, where the root is NOUN, ADJ, ADV, NUM, PRON, PROPN.
    :param data: list sample texts to be evaluated. Each sample text is a list of Sentence.
    """
    feature_measures = [_calc_rate_of_sentence_by_root(text, {'NOUN', 'ADJ', 'ADV', 'NUM', 'PRON', 'PROPN'}) for text in data]
    return feature_measures


def calc_number_of_words(data: list[list[Word]]) -> list[int]:
    """
    Calculates number of words in the text. The words are space separated tokens, that are not punctuation marks.
    :param data: list sample texts to be evaluated. Each sample text is a list of Word.
    """
    feature_measures = [len(text) for text in data]
    return feature_measures


def calc_pos_rate(data: list[list[Word]], **kwargs) -> Union[dict[str, list[float]], None]:
    """
    Calculates a rate of words of various part-of-speech in a given text.
    :param data: list sample texts to be evaluated. Each sample text is a list of Word.
    The text is also accepted as parsed (stanza.models.common.doc.Document)
    """
    kwargs = kwargs.get('arg', None)  # get real **kwargs passed into function as 'arg' argument
    pos_to_analyze = None
    if kwargs is not None:
        pos_to_analyze = kwargs.get('pos', pos_to_analyze)
    if pos_to_analyze is None:
        return None
    pos_rate = []
    for text in data:
        rate = defaultdict(int)
        for w in text:
            rate[w.pos] += 1
        rate = {wp: rate[wp] / len(text) for wp in rate}
        pos_rate.append(rate)
    feature_measures = {p: [sample.get(p, 0) for sample in pos_rate] for p in pos_to_analyze}
    return feature_measures


def calc_root_dep_counts(data: list[list[Sentence]]) -> Union[list[float], list[np.ndarray]]:
    """
    Calculates a mean number of words dependent on the root in each sentence of the text.
    :param data: list sample texts to be evaluated. Each sample text is a list of Sentence.
    The dependent tokens with one of the following PoS are skipped:
    AUX (will), PUNCT (.), DET (a), PART (n't')
    """
    feature_measures = []
    for text in data:
        text_values = []
        for sentence in text:
            root = [w for w in sentence.words if w.head == 0]
            root_id = root[0].id
            dep_words = [w for w in sentence.words if w.head == root_id]
            dep_words = [w for w in dep_words if w.upos not in ('AUX', 'PUNCT', 'DET', 'PART')]
            text_values.append(len(dep_words))
        feature_measures.append(np.mean(text_values))
    return feature_measures


def calc_root_pos_distribution(data: list[list[Sentence]]) -> dict[str, float]:
    """
    Get a distribution of part-of-speech of sentence roots within the text.
    :param data: list sample texts to be evaluated. Each sample text is a list of Sentence.
    """
    pos_rate = []
    pos_names = set()
    for text in data:
        s_root = [_get_root_pos(sentence) for sentence in text]
        s_root_rate = Counter(s_root)
        rate = {wp: s_root_rate[wp] / len(text) for wp in s_root_rate}
        pos_names.update(list(Counter.keys()))
        pos_rate.append(rate)
    feature_measures = {p: [sample.get(p, 0) for sample in pos_rate] for p in pos_names}
    return feature_measures


def calc_sentence_length(data: list[list[Sentence]]) -> Union[list[float], list[np.ndarray]]:
    """
    Calculates a mean sentence length in a given text.
    :param data: list sample texts to be evaluated. Each sample text is a list of Sentence.
    """
    feature_measures = []
    for text in data:
        text_value = [len([w for w in s.words if w.pos != 'PUNCT']) for s in text]
        feature_measures.append(np.mean(text_value))
    return feature_measures


def calc_size_of_vocabulary(data: list[list[Word]]) -> list[int]:
    """
    Calculates size of the vocabulary within a text.
    :param data: list sample texts to be evaluated. Each sample text is a list of Word.
    """
    feature_measures = []
    for word_list in data:
        n_lemmas = len(set([w.lemma for w in word_list]))
        feature_measures.append(n_lemmas)
    return feature_measures


def calc_type_token_ratio(data: list[list[Word]]) -> list[float]:
    """
    Calculates type-token-ratio of the text.
    :param data: list sample texts to be evaluated. Each sample text is a list of Word.
    """
    feature_measures = []
    for text in data:
        lemmas = {w.lemma for w in text}
        ttr_value = len(lemmas) / len(text)
        feature_measures.append(ttr_value)  # feature_measures.append(n_words / n_lemmas)
    return feature_measures


def calc_verb_per_sentence(data: list[list[Sentence]]) -> Union[list[float], list[np.ndarray]]:
    """
    Calculates a mean number of verbs per sentence in a given text.
    :param data: list sample texts to be evaluated. Each sample text is a list of Sentence.
    """
    feature_measures = []
    for text in data:
        n_verbs = [len([w for w in s.words if w.upos == 'VERB']) for s in text]
        feature_measures.append(np.mean(n_verbs))
    return feature_measures


def calc_vocabulary_ranks_difference(texts: Tuple[list[list[Word]], list[list[Word]]], **kwargs) -> Union[list[float], None]:
    """
    Calculates Jensen-Shannon divergence for the vocabulary distributions of two texts.
    Words that appear in random sample less than 'threshold' times are not taken into account.
    :param texts: (a, b) first text (A) and second text (B).
    :param threshold: frequency threshold (default is 2).
    The number of word tokens to be compared is the minimum lengths of text A and B.
    """
    threshold = 2  # default frequency threshold.
    kwargs = kwargs.get('arg', None)  # get real **kwargs passed into function as 'arg' argument
    if kwargs is not None:
        threshold = kwargs.get('threshold', threshold)
    feature_measures = []
    texts_a = texts[0]
    texts_b = texts[1]
    for i in range(len(texts_a)):
        words_a = [w.lemma for w in texts_a[i]]
        words_b = [w.lemma for w in texts_b[i]]
        a_rank_distribution = _calc_ranked_distribution(words_a, threshold)
        b_rank_distribution = _calc_ranked_distribution(words_b, threshold)

        min_len = min(len(a_rank_distribution), len(b_rank_distribution))
        a_rank_distribution = a_rank_distribution[:min_len]
        b_rank_distribution = b_rank_distribution[:min_len]
        distance = jensenshannon(a_rank_distribution, b_rank_distribution)
        feature_measures.append(distance)
    return feature_measures


def calc_word_length(data: list[list[Word]]) -> Union[list[float], list[np.ndarray]]:
    """
    Calculates mean length of words in the text.
    :param data: list sample texts to be evaluated. Each sample text is a list of Word.
    """
    feature_measures = []
    for text in data:
        text_value = [len(w.text) for w in text]
        feature_measures.append(np.mean(text_value))
    return feature_measures


def _calc_rate_of_sentence_by_root(text: list[Sentence], roots: set[str]) -> float:
    s_target = [sentence for sentence in text if len(sentence.words) > 0 if _get_root_pos(sentence) in roots]
    s_target_rate = len(s_target) / len(text)
    return s_target_rate


def _get_root_pos(sentence: Sentence) -> str:
    """
    Get a part-of-speech of sentence root.
    :param sentence: parsed sentence (by means of stanza toolkit).
    """
    root = [w for w in sentence.words if w.head == 0]
    root_pos = root[0].pos
    return root_pos


def _calc_ranked_distribution(list_of_tokens: list[str], freq_threshold: int = 2) -> list[int]:
    sample_distr = Counter(list_of_tokens)
    rank_distr = Counter([v for v in sample_distr.values() if v >= freq_threshold])
    rank_distr_sorted = sorted(rank_distr.values(), reverse=True)
    return rank_distr_sorted
