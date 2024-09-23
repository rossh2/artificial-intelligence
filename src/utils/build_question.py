import re

import pandas as pd

from utils.bigrams import bigram_to_string, string_to_bigram
from utils.io import read_words, read_bigrams

# Use 'an' instead of 'a' for count nouns if the first word matches the following
# 'useful' is an exception
AN_REGEX = re.compile(r'^[aeiou](?!seful)')

SINGULAR_VERB = 'is'
PLURAL_VERB = 'are'
# Include space after determiner
CONS_DET = 'a '
VOWEL_DET = 'an '
CONTEXT_QUESTION_FORMAT_STRING = 'In this setting, {verb} the {bigram} still {noun_det}{noun}?'
ISA_QUESTION_FORMAT_STRING = '{verb} {bigram_det}{bigram} still {noun_det}{noun}?'
ISA_FILLER_QUESTION_FORMAT_STRING = '{verb} {bigram_det}{bigram} still {target_phrase}?'


def add_question_column(bigrams: pd.DataFrame, mass_count: pd.DataFrame,
                        question_format_string: str, context: bool = False,
                        column_name: str = 'Question') -> pd.DataFrame:
    bigrams = add_mass_count_column(bigrams, mass_count)
    add_morphology_columns(bigrams)
    if context:
        bigrams[column_name] = bigrams.apply(lambda x: build_context_question(format_string=question_format_string,
                                                                              bigram=x.Bigram,
                                                                              noun=x.Noun,
                                                                              context=x.Context,
                                                                              noun_det=x.Noun_Det,
                                                                              verb=x.Verb)
                                             , axis=1)
    else:
        bigrams[column_name] = bigrams.apply(lambda x: build_bigram_question(format_string=question_format_string,
                                                                             bigram=x.Bigram,
                                                                             noun=x.Noun,
                                                                             bigram_det=x.Bigram_Det,
                                                                             noun_det=x.Noun_Det,
                                                                             verb=x.Verb)
                                             , axis=1)

    # Tidy up added columns
    bigrams.drop(columns=['MassCount', 'Noun_Det', 'Bigram_Det', 'Verb'], inplace=True)
    return bigrams


def build_bigram_question(format_string: str, bigram: str, noun: str, bigram_det: str, noun_det: str,
                          verb: str = SINGULAR_VERB) -> str:
    formatted = format_string.format(verb=verb, bigram=bigram, noun=noun, bigram_det=bigram_det, noun_det=noun_det)
    # Capitalize first letter if not already (might be verb)
    formatted = formatted[0].upper() + formatted[1:]
    return formatted


def build_context_question(format_string: str, bigram: str, noun: str, context: str, noun_det: str,
                           verb: str = SINGULAR_VERB) -> str:
    formatted = format_string.format(verb=verb, bigram=bigram, noun=noun, context=context, noun_det=noun_det)
    # Capitalize first letter if not already (might be verb)
    formatted = formatted[0].upper() + formatted[1:]
    return formatted


def build_filler_question(format_string: str, bigram: str, bigram_det: str, target_phrase: str,
                          verb: str = SINGULAR_VERB) -> str:
    formatted = format_string.format(verb=verb, bigram=bigram, bigram_det=bigram_det, target_phrase=target_phrase)
    # Capitalize first letter if not already (might be verb)
    formatted = formatted[0].upper() + formatted[1:]
    return formatted


def add_mass_count_column(concat_bigrams: pd.DataFrame, mass_count: pd.DataFrame) -> pd.DataFrame:
    assert 'MassCount' in mass_count.columns and 'Noun' in mass_count.columns
    return pd.merge(concat_bigrams, mass_count, on='Noun')


def add_morphology_columns(all_data: pd.DataFrame) -> None:
    all_data['Bigram_Det'] = all_data.apply(lambda row: get_determiner(row['Bigram'], row['MassCount']), axis=1)
    all_data['Noun_Det'] = all_data.apply(lambda row: get_determiner(row['Noun'], row['MassCount']), axis=1)
    all_data['Verb'] = all_data.apply(lambda row: get_verb(row['MassCount']), axis=1)


def get_determiner(word: str, mass_count: str) -> str:
    if mass_count == 'Count' or mass_count == 'Mass/Count':
        # Include space after determiner
        return VOWEL_DET if AN_REGEX.match(word) else CONS_DET
    else:
        return ''


def get_verb(mass_count: str) -> str:
    if mass_count == 'Plural':
        return PLURAL_VERB
    else:
        return SINGULAR_VERB


def build_bigram_questions(bigram_path: str, adjectives_path: str, mass_count_path: str,
                           question_format_string: str, context: bool = False,
                           question_column_name: str = 'Question') -> pd.DataFrame:
    if context:
        bigrams_df = pd.read_csv(bigram_path)
        # Generated contexts contain empty rows at end due to batch size, delete them
        bigrams_df = bigrams_df[~bigrams_df['Bigram'].isna()]
        bigrams_df = expand_context_bigram_df(bigrams_df)
    else:
        bigrams_df = build_bigram_df_from_strings(bigram_path)

    target_adjectives = read_words(adjectives_path)
    bigrams_df = bigrams_df[bigrams_df['Adjective'].isin(target_adjectives)]

    mass_count_df = pd.read_csv(mass_count_path)
    bigrams_df = add_question_column(bigrams=bigrams_df, mass_count=mass_count_df,
                                     question_format_string=question_format_string,
                                     context=context, column_name=question_column_name)

    return bigrams_df


def build_bigram_df_from_strings(bigram_path: str) -> pd.DataFrame:
    bigrams = read_bigrams(bigram_path)
    adjectives, nouns = zip(*bigrams)
    bigrams = [bigram_to_string(bigram) for bigram in bigrams]
    bigrams_df = pd.DataFrame({
        'Adjective': adjectives,
        'Noun': nouns,
        'Bigram': bigrams
    })
    return bigrams_df


def expand_context_bigram_df(bigrams_df: pd.DataFrame) -> pd.DataFrame:
    bigrams_df["id"] = bigrams_df.index
    bigrams_df = pd.wide_to_long(bigrams_df, i='id', j='ContextBias', stubnames=['Context'], suffix=r'[a-zA-Z0-9]+')
    bigrams_df = bigrams_df.reset_index(level=['ContextBias'])  # Convert to column
    bigrams_df = bigrams_df.reset_index(level=['id'], drop=True)  # Drop temporary index
    bigrams_df['Adjective'], bigrams_df['Noun'] = zip(*bigrams_df['Bigram'].apply(string_to_bigram))

    # Drop any lines where the context is empty
    bigrams_df = bigrams_df[~bigrams_df['Context'].isna()]

    return bigrams_df
