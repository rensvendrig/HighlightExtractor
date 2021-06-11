import numpy as np
from encoder import Encoder
from numpy import linalg as LA
import logging
import math
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import cluster
from collections import defaultdict, Counter
from yellowbrick.cluster import KElbowVisualizer, kelbow_visualizer
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter


class Ranker(object):

    def rank(self, doc_cluster, query, requirements, model, length_limit, diversity, do_unsuperised, bulk_parameter,requirement_weight=0.35,query_weight1 = 0.30, query_weight2=0.30):
        sort = False
        #ranked_sent_list = lex_rank_documents(doc_cluster, query)
        ranked_sent_list, phrase_embeddings = self.bert_rank_documents(doc_cluster, query, requirements, model, length_limit, diversity, do_unsuperised, bulk_parameter,requirement_weight,query_weight1, query_weight2)
        # the bigger the distance the better
        sorted_sents = dict(sorted(ranked_sent_list.items(), key=lambda item: -item[1])) if sort else ranked_sent_list
        return sorted_sents, phrase_embeddings

    def bert_rank_documents(self, sent_list, query, requirements, model, length_limit, diversity, algorithm, bulk_parameter,requirement_weight,query_weight1, query_weight2):

        # model_class, tokenizer_class, pretrained_weights = (transformers.DistilBertModel, transformers.DistilBertTokenizer, 'distilbert-base-uncased')
        # tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        # model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

        # Check if were dealing with seperate abstracts with sentences as lists (nested list) or all lists thrown together
        if any(isinstance(i, list) for i in sent_list):
            embeddings_of_abstracts = []
            abstracts_list = sent_list
            sent_list = []
            sent_belong_to_abs_idxs = []
            for abs_idx, abstract_sents in enumerate(abstracts_list):
                abstract_embedding = model.encode(' '.join(abstract_sents))
                embeddings_of_abstracts.append(abstract_embedding)
                sent_list.extend(abstract_sents)
                sent_belong_to_abs_idxs.extend(len(abstract_sents) * [abs_idx])
            doc_embedding = np.average(embeddings_of_abstracts, axis=0)

        else:
            # This is the way to go, not make embeddings for individual sentences, as it loses it semantic context.
            # You rather want to keep the abstract as a whole, combined with the query, and then check every sentence
            context = ' '.join(sent_list)
            doc_embedding = model.encode([context])

        # ngrams = Preprocess().sent_list_to_subsentences(sent_list)
        # input_context = "[CLS]" + context + "[SEP]" + query + "[SEP]"
        # doc_query_embedding = model.encode([input_context])
        # ngrams_embeddings = model.encode(ngrams)

        sentence_embeddings = model.encode(sent_list)
        query_embedding = model.encode([query])
        no_requirements = False
        if requirements == [] or no_requirements:
            query_weight = query_weight1
            doc_query_embedding = doc_embedding *(1-query_weight) + query_embedding *query_weight
        else:
            query_weight = query_weight2
            requirements_weight = requirement_weight
            # indi_req_weight = 0.25/float(len(requirements))

            cos_sum = cosine_similarity(sentence_embeddings, model.encode(requirements))
            coslist = []
            for sen_i, sentence_list in enumerate(cos_sum):
                for cos_i, cos in enumerate(sentence_list):
                    if cos > 0.5:
                        coslist.append(cos_i)
            new_requirements = list(np.array(requirements)[list(set(coslist))])
            if new_requirements == []:
                requirement_embedding = model.encode(', '.join(requirements))
            else:
                requirement_embedding = model.encode(', '.join(new_requirements))

            # requirements = ', '.join(requirements) #for now, join all the requirements with comma so they are all entailed in one embedding
            requirement_weight_times_embedding = requirements_weight * requirement_embedding

            # if len(requirements) > 1:
            #     for requirement in requirements[1:]:
            #         requirement_weight_times_embedding += model.encode(requirement) * indi_req_weight

            doc_query_embedding = doc_embedding * (1 - query_weight- requirements_weight) + query_embedding * query_weight + requirement_weight_times_embedding
        # ranked_sents_dict = dict(list(zip(ngrams, distances[0].tolist())))
        phrase_embeddings = sentence_embeddings
        phrase_list = sent_list

        if algorithm == 'kmeans':
            phrase_list = self.kmeans_before_ranking(phrase_embeddings, phrase_list, length_limit)
            ranked_sents_dict = {k: 1 for v, k in enumerate(phrase_list)}
        elif algorithm == 'lex':
            ranked_sents_dict = Ranker().lex_rank_documents(phrase_list, query, phrase_embeddings, doc_query_embedding, len(set(sent_belong_to_abs_idxs)))
        else:
            # ranked_sents_dict2 = Ranker().lex_rank_documents(phrase_list, query, phrase_embeddings, doc_embedding)
            ranked_sents_dict1 = Ranker().mmr(doc_query_embedding, phrase_embeddings, phrase_list,sent_belong_to_abs_idxs, diversity, length_limit, bulk_parameter)
            # ranked_sents_dict = self.add_dict_values(ranked_sents_dict1, ranked_sents_dict2)
            # ranked_sents_dict1[list(ranked_sents_dict2.keys())[0]] = int(list(ranked_sents_dict2.values())[0]) + int(list(ranked_sents_dict1.values())[0])
            ranked_sents_dict = ranked_sents_dict1

        # ranked_sents_dict = self.kmeans_after_ranking(phrase_embeddings, sent_list, ranked_sents_dict, num_of_abstracts)

        return ranked_sents_dict, phrase_embeddings

    def mmr(self, doc_embedding, word_embeddings, sentences,sent_belong_to_abs_idxs, diversity, cutoff, bulk_parameter):
        if cutoff > len(sentences):
            cutoff = len(sentences)
        coverage_selection = 0
        results = {}
        coverages = []
        i = 0
        num_abstracts = len(set(sent_belong_to_abs_idxs))
        abstracts_idx = list(set(sent_belong_to_abs_idxs))

        result_sent_emb = []
        normal_mmr_method = False
        do_bulk_sen_feature = True

        # Extract similarity within words, and between words and the document
        sentence_doc_similarity = cosine_similarity(word_embeddings,doc_embedding)
        sentence_similarity = cosine_similarity(word_embeddings)
        # Initialize candidates and already choose best keyword/keyphrase
        highest_mmr_idx = np.argmax(sentence_doc_similarity)
        highlight_idx = [highest_mmr_idx]
        candidates_idx = [i for i in range(len(sentences)) if i != highlight_idx[0]]

        if normal_mmr_method:
            for _ in range(cutoff - 1):
                candidate_similarities = sentence_doc_similarity[candidates_idx, :]
                target_similarities = np.max(sentence_similarity[candidates_idx][:, highlight_idx], axis=1)
                if do_bulk_sen_feature:
                    temp_mmr = (1 - diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
                    mmr = temp_mmr * self.score_surrounding_sentences(sentence_doc_similarity, sent_belong_to_abs_idxs,
                                                                       highlight_idx, bulk_parameter)[candidates_idx, :]
                else:
                    mmr = (1 - diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
                highest_mmr_idx = candidates_idx[np.argmax(mmr)]
                highlight_idx.append(highest_mmr_idx)
                candidates_idx.remove(highest_mmr_idx)
            return {sentences[idx]: sentence_doc_similarity[idx][0] for idx in highlight_idx}
        else:
            results[sentences[highest_mmr_idx]] = sentence_doc_similarity[highest_mmr_idx][0]
            # result_sent_emb.append(word_embeddings[highest_mmr_idx])
            while True:
                i+= 1
                # Do surrounding_sentences_scoring
                # Extract similarities within candidates and
                # between candidates and selected keywords/phrases

                # if i > num_abstracts:
                #     new_candidates_idx = candidates_idx
                # else:
                #     if i !=1:
                #         try:
                #             abstracts_idx.pop(last_sent_chosen_abs_idx)
                #         except IndexError:
                #             continue
                #     new_candidates_idx = [i for i in range(len(sentences)) if i not in highlight_idx and sent_belong_to_abs_idxs[i] in abstracts_idx]
                #     if new_candidates_idx == []:
                #         new_candidates_idx = candidates_idx

                candidate_similarities = sentence_doc_similarity[candidates_idx, :]
                target_similarities = np.max(sentence_similarity[candidates_idx][:, highlight_idx], axis=1)

                if i > num_abstracts and do_bulk_sen_feature:
                    temp_mmr = (1 - diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
                    mmr = temp_mmr * self.score_surrounding_sentences(sentence_doc_similarity, sent_belong_to_abs_idxs, highlight_idx, bulk_parameter)[candidates_idx, :]
                else:
                    mmr = (1 - diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)

                if mmr.size == 0:
                    return results
                highest_mmr_idx = candidates_idx[np.argmax(mmr)]

                new_coverage_selection, result_sent_emb = self.selection_method(doc_embedding, highest_mmr_idx,
                                                                                result_sent_emb,  word_embeddings)
                # coverages.append(new_coverage_selection)
                # print(new_coverage_selection, 'with: ', sentences[highest_mmr_idx])
                # print(coverages.index(max(coverages)))
                if len(results.keys()) >= 9:
                    break
                if num_abstracts <= len(results.keys()):
                    if coverage_selection < new_coverage_selection:
                        coverage_selection = new_coverage_selection
                    else:
                        break
                else:
                    coverage_selection = new_coverage_selection

                results[sentences[highest_mmr_idx]] = sentence_doc_similarity[highest_mmr_idx][0]

                # last_sent_chosen_abs_idx = sent_belong_to_abs_idxs[highest_mmr_idx]
                # Update keywords & candidates
                highlight_idx.append(highest_mmr_idx)
                candidates_idx.remove(highest_mmr_idx)
            return results


    def selection_method(self, doc_embedding, highest_mmr_idx, result_sent_emb, word_embeddings):
        result_sent_emb.append(word_embeddings[highest_mmr_idx])
        if len(result_sent_emb) >= 2:
            result_sent_emb = np.average(result_sent_emb, axis=0)
        else:
            result_sent_emb = result_sent_emb[0]
        new_coverage_selection = cosine_similarity(result_sent_emb.reshape(1, -1), doc_embedding)[0][0]
        result_sent_emb = [result_sent_emb]
        return new_coverage_selection, result_sent_emb

    def score_surrounding_sentences(self, sentence_doc_similarity, sent_belong_to_abs_idxs,
                                    selected_highlight_positions, bulk_parameter):
        min_score = bulk_parameter
        current_score = list(sentence_doc_similarity.copy())
        new_score = [[1] for i in range(len(current_score))]
        newly_added_sent_idx = selected_highlight_positions[-1]
        for highlight_position in selected_highlight_positions:
            temp_score = []
            for sent_idx, similarity in enumerate(current_score):
                abs_idx_of_highlight = sent_belong_to_abs_idxs[highlight_position]
                # if the current sentence is in the same abstract as the highlight, then do scoring on that sentence.
                if sent_belong_to_abs_idxs[sent_idx] == abs_idx_of_highlight:
                    new_candidate_score = [self.surrounding_sentences_scoring(sent_idx, highlight_position,
                                                                              Counter(sent_belong_to_abs_idxs)[
                                                                                  abs_idx_of_highlight], min_score)]
                else:  # otherwise, just give it the minimal score.
                    new_candidate_score = [min_score]
                temp_score.append(new_candidate_score)

            new_score = np.array(new_score) * np.array(temp_score)

        return np.array([[np.float(c[0])] for c in new_score],
                        np.float32)

    def surrounding_sentences_scoring(self,position_of_sentence, selected_highlight_position, num_of_sentences, min_score):
        p = position_of_sentence
        h = selected_highlight_position
        M = num_of_sentences
        return max(min_score, np.exp(-abs(h - p) / (M ** 2)))
'''Method that creates clusters and selects the phrase that is closest to each cluster centroid point. 
K is determined by the elbow method. 
'''
def kmeans_before_ranking(self, phrase_embeddings, phrases, k):
    kmeans = cluster.KMeans(random_state=42, init='k-means++')
    use_elbow_method = False
    if k == 'elbow_method':
        use_elbow_method = True
    optimal_k = k
    if use_elbow_method:
        visualizer = kelbow_visualizer(kmeans, phrase_embeddings, k=(2, len(phrases)), timings=False, show=False)
        optimal_k = visualizer.elbow_value_

        if optimal_k == None:
            return phrases
    elif len(phrases) < optimal_k or optimal_k <= 2:
        return phrases

    kmeans = cluster.KMeans(n_clusters=optimal_k, random_state=42, init='k-means++')
    kmeans.fit(phrase_embeddings)
    centroids = kmeans.cluster_centers_

    closest_points, _ = pairwise_distances_argmin_min(centroids, phrase_embeddings, metric='cosine')
    closest_phrases = [phrases[closest_point] for closest_point in closest_points]
    closest_phrases_embedding = [phrase_embeddings[closest_point] for closest_point in closest_points]
    return closest_phrases


'''This method is clustering the phrases after ranking. When phrases that belong in the same cluster are higher than other 
sentences in other clusters, the other clusters are punished more on their score'''


# https://ai.intelligentonlinetools.com/ml/tag/k-means-clustering-example/
def kmeans_after_ranking(self, phrase_embeddings, phrases, ranked_dict, num_of_abstracts):
    if num_of_abstracts < 2:
        NUM_CLUSTERS = 2
    else:
        NUM_CLUSTERS = num_of_abstracts
    kmeans = cluster.KMeans(n_clusters=NUM_CLUSTERS, random_state=42, init='k-means++')

    if len(phrase_embeddings) < 2:
        return ranked_dict

    kmeans.fit(phrase_embeddings)

    labels = kmeans.labels_

    # This next chunk of code combines the previous score with the kmeans score
    exp_range = sorted(np.logspace(1.85, 2, num=len(phrases), dtype='int') / 100, reverse=True)

    joined_list = zip(labels, exp_range)
    d = defaultdict(list)

    for k, v in joined_list:
        d[k].append(v)
    for k, v in d.items():
        d[k] = np.prod(v)
    updated_ranked_dict = dict()
    for i, (sent, score) in enumerate(ranked_dict.items()):
        sentence_cluster = labels[i]
        updated_ranked_dict[sent] = score * d[sentence_cluster]

    # silhouette_score = metrics.silhouette_score(phrase_embeddings, labels, metric='euclidean')

    return dict(sorted(updated_ranked_dict.items(), key=lambda item: item[1], reverse=True))


def lex_rank_documents(self, sent_list, query, word_embeddings, doc_embedding, num_abstracts):
    query = [query]
    n = len(sent_list)
    results = {}
    coverage_selection = 0
    result_sent_emb = []
    sentence_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
    relation_weight = 0.2
    # Initialises the adjacency matrix
    adjacency_matrix = np.zeros((n, n))
    degree = np.zeros(n)

    for i, senti in enumerate(sent_list):
        for j, sentj in enumerate(sent_list):
            if i == j:
                adjacency_matrix[i][j] = 1
            elif i < j:
                tf_senti = Encoder().get_tf_dict(senti)
                tf_sentj = Encoder().get_tf_dict(sentj)
                adjacency_matrix[i][j] = self.compute_cos_similarity(tf_senti, tf_sentj)
                # adjacency_matrix[i][j] = cosine_similarity(phrase_embeddings[i].reshape(-1, 1),phrase_embeddings[j].reshape(-1, 1))
            else:
                adjacency_matrix[i][j] = adjacency_matrix[j][i]

            degree[i] += adjacency_matrix[i][j]

    for i in range(n):
        for j in range(n):
            adjacency_matrix[i][j] = adjacency_matrix[i][j] / degree[i]

    initial_scores = np.zeros(n)
    if query[0]:
        for i, senti in enumerate(sent_list):
            for sentj in query:
                tf_senti = Encoder().get_tf_dict(senti)
                tf_sentj = Encoder().get_tf_dict(sentj)
                sim = self.compute_cos_similarity(tf_senti, tf_sentj)
                # sim = cosine_similarity(phrase_embeddings[i], doc_embedding[0])
                initial_scores[i] += sim
        query_weight = initial_scores.sum()
        if query_weight != 0:
            initial_scores = initial_scores / query_weight
        else:
            initial_scores = np.ones(n, dtype=np.float64) / n
    else:
        initial_scores = np.ones(n, dtype=np.float64) / n
    scores = self._power_method(adjacency_matrix, initial_scores, relation_weight)
    sent_dict = dict(list(zip(sent_list, scores)))
    for sent_i, sentence in enumerate(sent_dict.keys()):
        results[sentence] = sentence_doc_similarity[sent_list.index(sentence)][0]
        new_coverage_selection, result_sent_emb = self.selection_method(doc_embedding, sent_list.index(sentence),
                                                                        result_sent_emb,
                                                                        word_embeddings)
        # coverages.append(new_coverage_selection)
        # print(coverages)
        # print(coverages.index(max(coverages)))
        if num_abstracts <= len(results.keys()):
            if coverage_selection < new_coverage_selection:
                coverage_selection = new_coverage_selection
            else:
                return dict(list(sent_dict.items())[:sent_i])  # break
        else:
            coverage_selection = new_coverage_selection

    return sent_dict


def _power_method(self, m, inital_scores, relation_weight):
    epsilon = 0.001
    max_iter = 10
    p = inital_scores.copy()
    inital_scores *= (1 - relation_weight)
    iter_number = 0
    while True:
        iter_number += 1
        new_p = inital_scores + m.T.dot(p) * relation_weight
        diff = LA.norm(new_p - p)
        p = new_p
        logging.debug('lexrank error: ' + str(diff))
        if diff < epsilon:
            break
        if iter_number >= max_iter: break
    logging.debug('lexrank converged after ' + str(iter_number) + ' iterations.')
    return p


def compute_cos_similarity(self, dict1, dict2):
    sum_common = 0.
    sum1 = 0
    for k, v1 in dict1.items():
        if k not in dict2: continue
        v2 = dict2[k]
        sum_common += v1 * v2
        sum1 += v1 * v1
    if sum_common == 0: return 0
    sum2 = sum([v * v for v in dict2.values()])
    m = math.sqrt(sum1 * sum2)
    return sum_common / m


def add_dict_values(self, dict1, dict2):
    dic = dict(Counter(dict1) + Counter(dict2))
    return dict(sorted(dic.items(), key=lambda item: -item[1]))

# def score_surrounding_sentences(self,sentence_doc_similarity, sent_belong_to_abs_idxs, selected_highlight_positions):
#     min_score = 0.85
#     current_score = list(sentence_doc_similarity.copy())
#     new_score = []
#     newly_added_sent_idx = selected_highlight_positions[-1]
#
#     for sent_idx, similarity in enumerate(current_score):
#         abs_idx_of_highlight = sent_belong_to_abs_idxs[newly_added_sent_idx]
#         # if the current sentence is in the same abstract as the highlight, then do scoring on that sentence.
#         if sent_belong_to_abs_idxs[sent_idx] == abs_idx_of_highlight:
#             new_candidate_score = [self.surrounding_sentences_scoring(sent_idx, newly_added_sent_idx,
#                                                                  Counter(sent_belong_to_abs_idxs)[
#                                                                      abs_idx_of_highlight], min_score)]
#         else:  # otherwise, just give it the minimal score.
#             new_candidate_score = [min_score]
#         new_score.append(new_candidate_score)
#
#     return np.array([[np.float(c[0])] for c in new_score],
#                     np.float32)