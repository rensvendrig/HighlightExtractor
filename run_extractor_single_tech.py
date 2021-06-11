import ast
import re
from sentence_transformers import SentenceTransformer
from extractor import Extractor


def end2end(doc_content_list, technology_name, requirements):
    bertmodel = 'stsb-bert-base'
    model = SentenceTransformer(bertmodel)

    # These hyperparameters came out of hyperparameter tuning with UCB
    bulk_parameter = 0.95
    diversity = 0.257
    requirement_weight = 0.35
    tech_weight = 0.25
    min_cossim_req_include = 0.5

    # If requirement is present, save it as a list. If not, save it as an empty list.
    requirements_for_tech = [] if isinstance(requirements, float) else requirements

    abs_with_sentences, sent_belong_to_abs_idxs, sentences = parse_doc_list(doc_content_list)

    extractor = Extractor(diversity=diversity, bulk_parameter=bulk_parameter, requirement_weight=requirement_weight,
                          tech_weight=tech_weight, min_cossim_req_include=min_cossim_req_include)
    highlights = extractor.rank(abs_with_sentences, technology_name, requirements_for_tech, model,
                                sent_belong_to_abs_idxs)

    concat_highlights = concatinate_sequential_highlights(list(highlights.keys()), abs_with_sentences)

    return concat_highlights


def parse_doc_list(doc_content_list):
    sent_belong_to_abs_idxs, sentences, abs_with_sentences = [], [], []
    for doc_idx, document in enumerate(doc_content_list):
        doc_sentences = sentence_tokenizer(document)
        abs_with_sentences.append(doc_sentences)
        sentences.extend(doc_sentences)
        sent_belong_to_abs_idxs.extend(len(doc_sentences) * [doc_idx])
    return abs_with_sentences, sent_belong_to_abs_idxs, sentences


def sentence_tokenizer(document):
    return [x.lstrip().rstrip().rstrip('.') + '.' for x in
            re.split(r'[\!\?\.]\s{0,1}(?=[A-Z])', document) if re.compile(r'\w+').search(x)]


def concatinate_sequential_highlights(highlights, abs_with_sentences):
    '''This method adds highlights to each other if they appear before/ after each other in the text.
    FE: If abstract is ['sent1. sent2. sent3. sent4.'] and the highlights are ['sent1.', 'sent3.','sent4.'] then the highlights are exported as follows:
    [sent1.,'sent3. sent4.']'''
    new_highlights = []
    index_highlights = len(highlights) * [0]
    highlights_index_lists = []
    for abs_idx, abstract in enumerate(abs_with_sentences):
        highlights_index_dict = {}
        for sent_idx, sentence in enumerate(abstract):
            if sentence in highlights:
                index_highlights[highlights.index(sentence)] = (abs_idx, sent_idx)
                highlights_index_dict[sent_idx] = highlights.index(sentence)

        highlights_index_lists.append(highlights_index_dict)

    if 0 in index_highlights:
        return highlights

    for idx_sent_list in highlights_index_lists:
        concat_highlight = ''
        a = 0
        for sent_idx, high_idx in idx_sent_list.items():
            if (sent_idx + 1) in idx_sent_list.keys():
                if a == 0:
                    concat_highlight = highlights[high_idx] + ' ' + highlights[idx_sent_list[sent_idx + 1]]
                else:
                    concat_highlight = concat_highlight + ' ' + highlights[idx_sent_list[sent_idx + 1]]
                a += 1
            else:
                if a == 0:
                    new_highlights.append(highlights[high_idx])
                else:
                    new_highlights.append(concat_highlight)
                    a = 0

    ordered_highlights = []
    for original_highlight in highlights:
        for new_highlight in new_highlights:
            if original_highlight in sentence_tokenizer(new_highlight):
                ordered_highlights.append(new_highlight)
                new_highlights.remove(new_highlight)

    return ordered_highlights


def main():
    # The below values of the variables are just an example.

    # Right now considering that the technology is of type list of strings
    # One string is one document/abstract, so this string can consist of multiple sentences.
    paper_abstracts = [
        'Application of biotic growth regulators (e.g. humic components) and appropriate mulches is recommended to improve turfgrass quality. However, limited research has investigated their effect on lawn establishment. To investigate the effect of humic acid and selected mulches on characteristics of Festuca arundinacea in its planting stage, a factorial experiment based on a completely randomized block design with three replications was performed. The first factor was mulch types including vermicompost, leaf compost, cow manure and sand (control) which were used to cover the seeds. The second factor was a humic acid solution (100 ml l-1) sprayed monthly over the period of the experiment. Plant height, fresh and dry weight of lawn clippings, photosynthetic index, leaf texture and overall turfgrass quality were measured. Spraying humic acid significantly improved the measured factors except the dry weight and photosynthetic index of the plants. Among the mulches, vermicompost provided better impressions on improving the characteristics of this turfgrass species including 48% increase in fresh weight, 18% increase in height, 48% increase in total quality and 10% reduction in leaf width of the turfgrass. This research can assist in developing knowledge for achieving better quality lawns in urban landscapes.',
        'Salinity is one of the main abiotic stress factors which limit the growth and productivity of plants, however, the nutritional status of plants is the first brick in the resistance wall against stresses. Therefore, a factorial experiment was undertaken to investigate effects of soil applied humic acid (0, 7, 14, 21 l.ha -1 ) and boron foliar spraying (0, 50, 100 ppm) and their interaction on growth and yield of melon plant under saline conditions. The results suggested that the treatments soil application of humic acid and the boron spraying successfully mitigated the deleterious effects of salt stress and influenced growth and yield of melon plant. Humic acid at 21 l.ha -1 or boron spray at 50 ppm exhibited an improvement in growth and yield of melon, in terms of plant length, plant fresh and dry mass, chlorophyll (SPAD), fruit mass, total yield, and also leaf nutrient content (N and K) and total soluble solids (TSS) of fruits, while reduced the sodium content of leaves. The combined treatment of humic acid at 21 l.ha -1 and boron spraying at 50 ppm was found to be more effective for the melon plant to improving growth performance and the crop yield by 21% as compared with the control group under saline conditions.']

    # Right now considering that the technology is of type string
    techname = 'Humic substances'

    # Right now considering that the requirements is of type list of strings
    # Requirements can also be NaN! I will have the type float.
    requirements = ['Pre-harvest', 'Spray application', 'No fertiliser']

    highlights = end2end(paper_abstracts, techname, requirements)

    return highlights


if __name__ == '__main__':
    main()