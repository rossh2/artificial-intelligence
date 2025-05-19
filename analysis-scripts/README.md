# R Data Analysis Scripts

This directory contains the R scripts for processing the raw human experiment results, as well as doing statistics over the human and LLM experiment results.

The corresponding raw experiment results can be found in `results/human` and `results/llm` respectively (see instructions in the main README for unzipping). 

These R scripts are provided as-is for reference only. Unlike the Python code, they have not been cleaned up for public consumption, and include many analysis steps and functions that were not used in the final analysis. They may need modification in order to run in your environment. 

Most scripts should be standalone (should declare all the variables needed to run) but several scripts refer to CSV files that are dataframes exported from other scripts. In general, this is happening whenever the filename referenced is in the same directory as the script, e.g. `read.csv("isa_context_combined_target.csv")`. Look for the corresponding `write.csv` line in another script. It is possible that some scripts use variables declared in other scripts instead of importing a dataframe correctly.

If you have any questions about replicating this analysis, please reach out to the first author. An up-to-date email address can be found on her website.

## Contents

### Human data analysis

- `FilteringAnalysis.R` Processing and statistics for the first half of the bigram filtering experiment (Exp. 1 in the ELM paper)
- `Filtering2Analysis.R` Processing and statistics for the second half of the bigram filtering experiment, involving the "analogy" bigrams (Exp. 1 in the ELM paper)
- `IsanANanN_Analysis.R`  Processing and statistics for the first half of the no-context "Is an AN an N?" experiment (Exp. 2 in the ELM paper)
- `IsanANanN2_Analysis.R`  Processing and statistics for the second half of the no-context "Is an AN an N?" experiment, involving the "analogy" bigrams (Exp. 2 in the ELM paper)
- `ContextAnalysis.R` - Processing and statistics for the first half of human rating in context experiment (Exp. 3 in the ELM paper)
- `Context2Analysis.R` - Processing and statistics for the second half of human rating in context experiment (Exp. 3 in the ELM paper)
- `AnalogyPromptingAnalysis.R` - Processing and statistics for human analogical reasoning experiment (SCiL paper), as well as correlation between analogy model results and human results from original experiment
- `FakeAnalysis.R` Processing and statistics for the experiment on _fake_ and _real_ targeting k-properties in my dissertation
- `AnalysisUtils.R` - Functions to support human data processing and plot generation

### LLM data analysis

- `LMContextAnalysis.R` - Processing and statistics for the LLM context experiment (Exp. 1 in the GenBench paper)
- `LM_ISA_Analysis.R` - Processing and statistics for the LLM no-context experiment (Exp. 2 / Methods 1 and 2 in the GenBench paper)
- `LMGeneratedContextAnalysis.R` - Processing and statistics for the LLM context generation experiment (Exp. 3 / Method 3 in the GenBench paper)
- `LM_Analysis_Utils.R` Functions to support LLM data processing and plot generation

Some LLM responses are referenced in these files but were not used in the final analysis; this data is not provided in this repository but can be replicated if need be following the steps in the main README.

## Useful abbreviations

- "ISA" generally refers to the "Is an AN an N?" experiment in the ELM paper.
- "Analogy bigrams" refer to the second half of the bigrams which were added for later use in the analogy prompting experiment. Filtering or "ISA" experiments that are labeled as "with analogy" refer to the second version of these experiments that incorporate this second set of bigrams, and do not actually involve any analogy.
- Filenames involving "12" are capped at max 12 ratings/bigram and also usually restricted to the 12 target adjectives. This is necessary because the ISA experiment run on PCIbex sometimes produced more than 12 ratings per bigram due to technical errors. 