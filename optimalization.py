from rouge_score import rouge_scorer
import numpy as np
import ast
import scipy
from preprocess import Preprocess
from sklearn.metrics.pairwise import cosine_similarity
from bayes_opt import BayesianOptimization

class Evaluator(object):

    def optimization_function(self, doc_content_list, query, human_content_list, num_abstracts, requirements):

        for index, abstracts_of_tech in enumerate(doc_content_list):
            sentences = []
            abs_with_sentences = []
            requirements_for_tech = [] if isinstance(requirements[index], float) else ast.literal_eval(
                requirements[index])
            for document in ast.literal_eval(abstracts_of_tech):
                abs_with_sentences.append(sentence_tokenizer(document))
                sentences.extend(sentence_tokenizer(document))
                # sentences.extend(bulk_sentences(document))
            # summary, phrase_embeddings = summarize(abs_with_sentences, query[index], len(gold_summaries[index]), model, diversity, algorithm)
            summary, phrase_embeddings = summarize(abs_with_sentences, query[index], requirements_for_tech,
                                                   len(sentences),
                                                   model, diversity, algorithm)
            summary_sentences = list(summary.keys())
            pred_summaries.append(summary_sentences)


            iteration += 1
            if iteration in [(num_techs // 4) * 1, (num_techs // 4) * 2, (num_techs // 4) * 3]:
                print('Num of tech summarized: {}'.format(iteration))

            type_scores = ['rouge1']
            rouge_scores = Evaluator().calc_rouge_scores([summary_sentences], [gold_summaries[index]], [query[index]],
                                                         type_scores)

    def sentence_tokenizer(document):
        # sentences += tokenizer.tokenize(document.strip())
        return [x.lstrip().rstrip().rstrip('.') + '.' for x in
                re.split(r'[\!\?\.]\s{0,1}(?=[A-Z])', document) if re.compile(r'\w+').search(x)]

    '''This is the method on a list of  [(generated summary, gold summary), ] level'''
    def calc_rouge_scores(self, pred_summaries, gold_summaries, technologies,
                          keys=['rouge1'], use_stemmer=False):

        all_scores = {}
        for index, (pred_summary, gold_summary) in enumerate(zip(pred_summaries, gold_summaries)):
            dict_scores = self.calc_rouge_score(gold_summary, keys, pred_summary, use_stemmer)

            all_scores[technologies[index]] = dict_scores

        return self.display_average_rouge_score(all_scores, keys)

    '''This is the rouge method on one (generated summary, gold summary) level'''
    def calc_rouge_score(self, gold_summary, keys, pred_summary, use_stemmer):
        # Calculate rouge scores
        scorer = rouge_scorer.RougeScorer(keys, use_stemmer=use_stemmer)
        n = len(pred_summary)
        scores = [scorer.score(' '.join(pred_summary), ' '.join(gold_summary))]
        # create dict
        dict_scores = {}
        for key in keys:
            dict_scores.update({key: {}})
        # populate dict
        for key in keys:
            precision_list = [scores[j][key][0] for j in range(len(scores))]
            recall_list = [scores[j][key][1] for j in range(len(scores))]
            f1_list = [scores[j][key][2] for j in range(len(scores))]

            precision = np.mean(precision_list)
            recall = np.mean(recall_list)
            f1 = np.mean(f1_list)

            dict_results = {'recall': recall, 'precision': precision, 'f1': f1}

            dict_scores[key] = dict_results
        return dict_scores

    def display_average_rouge_score(self, all_scores, keys):
        all_precision_scores = []
        all_recall_scores = []
        all_f1_scores = []
        all_scores_with_key = {}
        for key in keys:
            for tech, scores in all_scores.items():
                all_precision_scores.append(scores[key]['precision'])
                all_recall_scores.append(scores[key]['recall'])
                all_f1_scores.append(scores[key]['f1'])

            all_scores_with_key[key] = (round(np.mean(all_precision_scores) * 100, 2), round(np.mean(all_recall_scores) * 100, 2),
             round(np.mean(all_f1_scores) * 100, 2))

        return all_scores_with_key


    '''Returns the mean of the mean distance between the ranks of each sentence of all the gold and pred summaries '''
    def distance_scores(self, pred_summaries, gold_summaries, technologies):
        all_scores = {}
        macro_distance_w_outliers = []
        macro_distance_wout_outliers = []
        for index, (pred_summary, gold_summary) in enumerate(zip(pred_summaries, gold_summaries)):
            micro_w_outliers, micro_wout_outliers = self.distance_score(gold_summary, pred_summary)
            macro_distance_wout_outliers.append(micro_wout_outliers)
            macro_distance_w_outliers.append(micro_w_outliers)
            all_scores[technologies[index]] = (micro_wout_outliers, micro_w_outliers)
        return np.mean(macro_distance_wout_outliers), np.mean(macro_distance_w_outliers)

    '''Returns the mean of the distance between the ranks of each sentence in the gold and pred summaries '''
    def distance_score(self, gold_summary, pred_summary):
        copied_pred_summary = pred_summary.copy()
        copied_gold_summary = gold_summary.copy()
        all_scores = []
        micro_distance = []
        for gold_idx, gold_sentence in enumerate(copied_gold_summary):
            pred_idx, pred_sentence = self.which_sent_is_same(gold_sentence, copied_pred_summary)
            all_scores.append((gold_idx, pred_idx)) #right now we are not using the all_score list
            micro_distance.append(abs(gold_idx - pred_idx))
            if copied_pred_summary != []:
                copied_pred_summary.pop(pred_idx)
            else:
                break
        return np.mean(micro_distance), np.mean(self.reject_outliers(micro_distance))

    def reject_outliers(self, data, m=2.):
        data = np.array(data)
        if len(data) <= 1:
            return data.tolist()
        d = np.abs(data - np.median(data))
        mdev = np.median(d)
        s = d / mdev if mdev else 0.
        return data[s < m].tolist()

    '''Method that looks at how many ngrams of the predicted summary is in a gold sentence of the gold summary. 
    The sentence that has the highest score should be the same sentence as the gold sentence. '''
    def which_sent_is_same(self, gold_sentence, pred_summary):
        try:
            score = []
            for pred_idx, pred_sentence in enumerate(pred_summary):
                ngram_in_sent = 0
                splitted_pred_sentence = pred_sentence.split()
                if(len(splitted_pred_sentence) > 4): # skips sentences that only have 2 or less tokens.
                    ngrams = Preprocess().to_ngram(pred_sentence, 5)
                    if ngrams != []: # if ngrams = [], the ngrams of the sentence are bigger than the splitted up sentence. See log.
                        for ngram in ngrams:
                            if ngram in gold_sentence.lower():
                                ngram_in_sent +=1
                    if ngram_in_sent != 0:
                        score.append((ngram_in_sent, pred_idx, pred_sentence))

            def getKey(item):
                return item[0]
            sorted_score = sorted(score, key=getKey, reverse=True)
            if sorted_score == []:
                sorted_score = [[0, 0,'ERROR SENTENCE']]
            return sorted_score[0][1], sorted_score[0][2]
        except IndexError:
            print("Index Error on gold sentence:{} and score {}, summary {}".format(gold_sentence, sorted_score, pred_summary) )

    def included_scores(self, gold_summaries, pred_summaries):
        included_score = []
        for index, (pred_summary, gold_summary) in enumerate(zip(pred_summaries, gold_summaries)):
            included_score.append(self.included_score(gold_summary, pred_summary))
        return round(np.mean(included_score), 5)


    def included_score(self, gold_summary, pred_summary):
        pred_high_included = 0
        for pred_highlight in pred_summary:
            if self.is_sent_included(pred_highlight, gold_summary):
                pred_high_included+= 1
        if pred_high_included > 0:
            return pred_high_included/float(len(pred_summary))
        else:
            return 0

    '''Method that checks on ngrams whether a sentence is included in the gold summary.'''
    def is_sent_included(self, pred_sentence, gold_summary):
        ngramrange = (5, 5)
        try:
            score = []
            for gold_idx, gold_sentence in enumerate(gold_summary):
                ngram_in_sent = 0
                splitted_gold_sentence = gold_sentence.split()
                if (len(splitted_gold_sentence) >= ngramrange[0]):  # skips sentences that only have 2 or less tokens.
                    ngrams = Preprocess().to_ngram(gold_sentence, ngramrange[0])
                    if ngrams != []:  # if ngrams = [], the ngrams of the sentence are bigger than the splitted up sentence. See log.
                        for ngram in ngrams:
                            if ngram in pred_sentence:
                                return True


            if score == []:
                return False

        except IndexError:
            print("Index Error on pred sentence:{} and summary {}".format(pred_sentence,
                                                                                    gold_summary))
    def calculate_coverages(self, pred_summaries, gold_summaries):
        coverage_scores = []
        for index, (pred_summary, gold_summary) in enumerate(zip(pred_summaries, gold_summaries)):
            coverage_scores.append(self.calculate_coverage(pred_summary, gold_summary))
        return coverage_scores

    """Returns how many gold highlights are in the pred summary, how many gold high are semantically similar to high in pred, 
    percentage of these two compared to all the gold highlights, and only semantically similar gold percentage to entrire gold summary. """
    def calculate_coverage(self, sentences, sentence_embeddings, pred_summ, gold_summary):
        gold_high_included = 0
        semantically_sim_sent = 0
        pred_summary = pred_summ.copy()
        for gold_idx, gold_highlight in enumerate(gold_summary):
            if self.is_sent_included(gold_highlight, pred_summary):
                gold_high_included += 1
                pred_summary.pop(self.get_index(pred_summary, gold_highlight, False))
            elif self.is_sentence_semantically_similar(sentences, sentence_embeddings, gold_highlight, pred_summary):
                semantically_sim_sent += 1
                pred_summary.pop(self.get_index(pred_summary, gold_highlight, False))
        return gold_high_included, semantically_sim_sent, round((gold_high_included + semantically_sim_sent)/float(len(gold_summary)), 2), round(semantically_sim_sent/float(len(gold_summary)), 2)

    def is_sentence_semantically_similar(self, sentences, sentence_embeddings, gold_highlight, pred_summary, algo = 'cosine'):
        threshold = 0.80

        for pred_idx, pred_sentence in enumerate(pred_summary):
            if algo == 'cosine':
                gold_idx_in_sent = self.get_index(sentences, gold_highlight, False)
                pred_idx_in_sent = self.get_index(sentences, pred_sentence, True)
                if gold_idx_in_sent & pred_idx_in_sent:
                    score = cosine_similarity([sentence_embeddings[gold_idx_in_sent]], [sentence_embeddings[pred_idx_in_sent]])[0][0]
                    if score > threshold:
                        return True
            else:
                continue
                # kmeans to be implemented
        return False

    def get_index(self, sentences, highlight, is_predicted):
        if is_predicted: #if the highlight is a predicted highlights, its definitly in the gold
            return sentences.index(highlight)
        else: # if the highlight is gold, it may be a little bit altered
            ngrams = Preprocess().to_ngram(highlight, 5)
            for index, sentence in enumerate(sentences):
                if ngrams != []:  # if ngrams = [], the ngrams of the sentence are bigger than the splitted up sentence. See log.
                    for ngram in ngrams:
                        if ngram in sentence.lower():
                            return index
        return False

    def selection_evaluator_per_sum(self, pred_sum_len, gold_sum_len):
        # So if the gold has 4 highs and pred 3, then it will give a 1 (score will be 0.25)
        # If gold has 3 and pred 2, then it will be given a 0 (score will be 0.33)
        abs_distance = abs(pred_sum_len - gold_sum_len)
        if abs_distance/float(gold_sum_len) <= 0.25:
            return 1, abs_distance
        else:
            return 0, abs_distance

    def is_nth_highlight_in_gold(self, pred_summary, gold_summary):
        first = 0
        second = 0
        for pre_idx, pred_sentence in enumerate(pred_summary):
            if Evaluator().is_sent_included(pred_sentence, gold_summary):
                if pre_idx == 0:
                    first += 1
                elif pre_idx == 1:
                    second += 1

        return first, second