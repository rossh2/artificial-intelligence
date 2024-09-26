# Is artificial intelligence still intelligence?

This repository contains the data and some scripts for human and LLM experiments on privative adjectives, 
such as "Is artificial intelligence still intelligence?"

The first part of this work (results of experiments with humans) was presented at ELM 2024 under the title 
"_Fake reefs_ are sometimes _reefs_ and sometimes not, but are always compositional".
The second part of this work (results of experiments with LLMs) will be presented at GenBench 2024 - repository
to be updated soon with the code and results.

## Accessing the data

To prevent the data in this repository from being crawled for LLM pretraining, the bigrams, results and surveys 
(which contain the bigrams) are stored as password-protected ZIP files. Unzip each of the ZIP files to its respective
path to run the scripts in this repository or to view the data. 
The password for each ZIP file is `artificial intelligence data`, except without the spaces (all lowercase). 

After unzipping, your directory structure should look like this:

```
artificial-intelligence/
├─ bigrams/
│  ├─ adjectives.txt
│  ├─ ...
├─ results/
│  ├─ frequencies/
│  ├─ human/
│  ├─ .../
├─ src/
├─ surveys/
```

## Overview of the data

The bigrams and handwritten contexts, as well as associated metadata, are located in `bigrams`.

- The list of adjectives used in all experiments is located in `adjectives.txt`
- The list of nouns used in Experiment 1 (filtering) and Experiment 2 ("Is an AN an N?") is located in `experiment_nouns.txt`, one noun per line. 
The full set of nouns used for corpus counting and to calculate the (relative) frequencies is given in `all_nouns.txt`.
- The list of bigrams used in Experiment 2 ("Is an AN an N?") is located in `experiment_bigrams.txt` 
(adjective and noun are tab-separated on each line, `src/utils/io.py > read_bigrams()` parses this format).
- The bigrams and biased contexts used in Experiment 3 ("In this setting, is the AN an N?") are located in `adjective_contexts.csv`.
- The other files in `bigrams` are used for Qualtrics survey generation.

The `results` directory contains results for the C4 corpus frequency counting script and for the three experiments.

The `surveys` directory contains the output of the Qualtrics survey generation scripts for Experiment 1 and Experiment 2 (see below),
which can be imported into Qualtrics, as well as exports in QSF format of Experiment 3.

## Corpus frequency script

We also provide the script to calculate the frequency of each adjective-noun bigram in the C4 corpus, at
`src/frequencies`. To run this script, you will need to install `pandas` and the HuggingFace `datasets` package, as 
listed in `environment.yml`. 
This script is designed to be run using multiple processes to speed up performance, for example on a CPU cluster.
(Using 8 CPUs, it will still take well over 24h to count all the bigrams used in these experiments.)

Example usage:
```shell
python src/frequencies/check_c4_bigram_frequency.py --processors 8 --chunk_size 1024 --doc_limit 365000000 --doc_group_size 200 --save_interval 100 --download --adjectives bigrams/adjectives.txt --nouns bigrams/all_nouns.txt --out_dir output/counts/
```

## Qualtrics survey generation scripts

In addition to the bigrams and results, we also share the scripts we used to programmatically generate Qualtrics
surveys for this large number of bigrams. Please feel free to adapt these scripts for your own use.

The resulting `.txt` file follows the [Qualtrics Advanced TXT format](https://www.qualtrics.com/support/survey-platform/survey-module/survey-tools/import-and-export-surveys/#PreparingAnAdvancedFormatTXTFile)
and can be imported to Qualtrics. Note that aspects such as the survey flow, question randomization and marking questions
as required are not supported by this format, and have to be added manually.

## Paper

If you use this work, please cite our paper! The paper is currently accepted and pending publication; 
in the meantime, you can view it on [LingBuzz](https://lingbuzz.net/lingbuzz/008012).

```bibtex
@article{ross2024fake,
	title = {Fake reefs are sometimes reefs and sometimes not, but are always compositional},
	author = {Ross, Hayley and Kim, Najoung and Davidson, Kathryn},
	journaltitle = {Experiments in Linguistic Meaning},
	volume = {3},
	date = {2024},
}
```
