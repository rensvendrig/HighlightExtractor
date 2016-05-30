import ranker
import selector
import pipeline

def build_common_ranker(model_name='lexrank'):
    if model_name=='lexrank':
        lexrank=ranker.LexRank()
        return lexrank
    if model_name=='manifold':
        manifold=ranker.ManifoldRank()
        return manifold
    return None


def build_common_selector():
    measurement=selector.NgramMeasurement()
    sent_filter=selector.SummarySentenceFilter()
    my_selector=selector.Selector(sent_filter,measurement)

    return my_selector

def build_common_pipeline():
    my_ranker=build_common_ranker('lexrank')
    my_selector=build_common_selector()
    my_pipeline=pipeline.SummarizationPipeline(my_ranker,my_selector)
    return my_pipeline