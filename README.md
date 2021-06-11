# HighlightExtractor
Abstract: 
This thesis aims to design, implement, and evaluate an automatic query-based
multi-document highlight extraction pipeline for technology scouting leveraging
a SBERT transformer. The highlight extraction problem was handled as an
extractive summarization problem. The general background of automatic summarization,
as well as the domain setting at Findest, is introduced. A highlight
extraction pipeline is proposed including a novel bulk scoring feature and a dynamic
selection method. The contribution of each component within the pipeline
is tested. The pipeline was also tested against common baselines and state-ofthe-
art models. The concluding remarks state that BERT and SBERT show
valuable contributions to extractive summarization and the proposed pipeline
outperforms other models on the domain-specific dataset.

## usage
- example.py                      run this file to generate multiple summaries on the dataset without margins. Change the parameters in the main() to modify the dataset. 
- baseline_models.py              run this file and change the parameters to check the ROUGE scores of the internal components of the pipeline and the baseline and SOTA models.
- run_extractor_single_tech.py    run this file to test a single technology. This file is also used by Findest in their AI. 
