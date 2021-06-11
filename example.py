from PyTimer import Stopwatch
import pandas as pd
import ast
import re
from sentence_transformers import SentenceTransformer
from ranker import Extractor
from evaluator import Evaluator
import csv

def end2end_test(doc_content_list, query, human_content_list, requirements):
    model = 'stsb-bert-base'
    model = SentenceTransformer(model)
    gold_summaries = [ast.literal_eval(gold_summary) for gold_summary in human_content_list]
    evaluation_file = "./data/rouge_scores_pipeline.csv"
    sum_file = './data/predicted_summaries.csv'
    create_csv_file(evaluation_file, ["Model", "Selection", "Diversity", "Rouge Score", "Included Score"])

    all_output = dict()
    pred_summaries = []
    num_techs = len(doc_content_list)
    iteration = 0
    first_lines = True

    # These hyperparameters came out of hyperparameter tuning with UCB
    bulk_parameter = 0.95
    diversity = 0.26
    requirement_weight = 0.35
    tech_weight = 0.25
    min_cossim_req_include = 0.5

    for index, abstracts_of_tech in enumerate(doc_content_list):
        abs_with_sentences, sent_belong_to_abs_idxs, sentences = [], [], []
        requirements_for_tech = [] if isinstance(requirements[index], float) else ast.literal_eval(requirements[index])
        for doc_idx, document in enumerate(ast.literal_eval(abstracts_of_tech)):
            doc_sentences = sentence_tokenizer(document)
            abs_with_sentences.append(doc_sentences)
            sentences.extend(doc_sentences)
            sent_belong_to_abs_idxs.extend(len(doc_sentences) * [doc_idx])

        summary_sentences, phrase_embeddings = summarize(abs_with_sentences, query[index], requirements_for_tech,
                                               model, sent_belong_to_abs_idxs, diversity, bulk_parameter, requirement_weight, tech_weight, min_cossim_req_include)
        pred_summaries.append(summary_sentences)

        if first_lines:
            all_output[query[index]] = dict()
            first_lines = False
        all_output[query[index]] = [summary_sentences]

        iteration += 1
        if iteration in [(num_techs // 4)*1, (num_techs // 4)*2, (num_techs // 4)*3]:
            print('Num of tech summarized: {}'.format(iteration))

    type_scores = ['rouge1', 'rouge2', 'rougeL']
    rouge_scores = Evaluator().calc_rouge_scores(pred_summaries, gold_summaries, query, type_scores)

    for type_score, rouge_score in rouge_scores.items():
        write_line_to_csv(evaluation_file, [type_score, rouge_score[0],rouge_score[1],rouge_score[2]])
        print('Mean {} score precision: {}\t recall: {}\t f1: {}\t '.format(type_score, rouge_score[0], rouge_score[1],
                                                                            rouge_score[2]))

    print_generated_summs(all_output, sum_file, gold_summaries, requirements)

    return


def print_generated_summs(all_output, sum_file, gold_summaries, requirements):
    create_csv_file(sum_file, ['index', 'summary'])
    for tech_idx, (key, values) in enumerate(all_output.items()):
        requirements_for_tech = [] if isinstance(requirements[tech_idx], float) else ast.literal_eval(requirements[tech_idx])
        write_line_to_csv(sum_file, ['\n' + 'Technology: ' + str(key) + ' Num highs: '+ str(len(gold_summaries[tech_idx]))  + '\t\t Req: '+ str(', '.join(requirements_for_tech))])
        for i, sent in enumerate(all_output[key][0]):
            if Evaluator().is_sent_included(sent, gold_summaries[tech_idx]):
                write_line_to_csv(sum_file, ['#', round(1, 3), '\t' + sent])
            else:
                write_line_to_csv(sum_file, [i, round(1, 3), '\t' + sent])
        write_line_to_csv(sum_file, ['\n'])


def print_generated_summary(index, query, summary):
    print("Technology: ", query[index])
    for sent, saliency in summary.items():
        print(saliency, '\t', sent)
    print('\n')

def sentence_tokenizer(document):
    return [x.lstrip().rstrip().rstrip('.') + '.' for x in
            re.split(r'[\!\?\.]\s{0,1}(?=[A-Z])', document) if re.compile(r'\w+').search(x)]

def create_csv_file(file, headers):
    # this creates a csv in the same directory folder
    with open(file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers)

def write_line_to_csv(file, data):
    with open(file, 'a+', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(data)

def summarize(abs_with_sentences,technology_name, requirements_for_tech, model, sent_belong_to_abs_idxs, diversity, bulk_parameter, requirement_weight, tech_weight, min_cossim_req_include):
    extractor = Extractor(diversity = diversity, bulk_parameter = bulk_parameter, requirement_weight = requirement_weight, tech_weight = tech_weight, min_cossim_req_include = min_cossim_req_include)
    highlights, phrase_embeddings = extractor.rank(abs_with_sentences, technology_name, requirements_for_tech, model, sent_belong_to_abs_idxs, 'TechReqBulkSel')
    return highlights, phrase_embeddings

def main():
    stopwatch=Stopwatch()
    stopwatch.start()

    df = pd.read_csv('data/new_test_set_wout_margins.csv')[700:702]

    end2end_test(df['PaperAbstract'].tolist(),df['TechName'].tolist(),df['Highlight'].tolist(), df['Requirements'].tolist())

    stopwatch.end_and_show('Summarizer')
    return


if __name__=='__main__':
    main()
