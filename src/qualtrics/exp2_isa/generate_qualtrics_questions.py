from typing import List, Tuple

import pandas as pd

from utils.bigrams import bigram_to_string
from utils.build_question import build_bigram_question, get_determiner, ISA_QUESTION_FORMAT_STRING, \
    build_filler_question, ISA_FILLER_QUESTION_FORMAT_STRING, get_verb
from qualtrics.qualtrics import QualtricsSurveyGenerator


class IsaSurveyGenerator(QualtricsSurveyGenerator):
    PAGE_BREAK_BETWEEN_BIGRAMS = True
    ONE_BLOCK_PER_GROUP_WITH_PAGE_BREAKS = True
    ONE_BIGRAM_PER_BLOCK = False
    SHUFFLE_BIGRAMS_WITHIN_BLOCKS = True

    TRAINING_BIGRAM_PATH = '../../../bigrams/isa_training_bigrams.json'
    FILLER_PATH = '.../../../bigrams/isa_filler_bigrams.csv'
    SURVEY_OUT_DIR = f'../../../surveys'

    QUESTION_PHRASING = ISA_QUESTION_FORMAT_STRING
    FILLER_QUESTION_PHRASING = ISA_FILLER_QUESTION_FORMAT_STRING
    FILLERS_FROM_TEMPLATE = False
    QUESTION_TYPE = 'Question:MC:SingleAnswer:Horizontal'
    MULTIPLE_CHOICE_ANSWERS: List[str] = [
        'Definitely not', 'Probably not', 'Unsure', 'Probably yes', 'Definitely yes'
    ]

    TRAINING_INTRO = 'Thank you for taking part!<br><br>' \
                     'In each question, you will be given an adjective-noun combination like <b>wooden potato</b> ' \
                     'and asked a question about it.<br><br>' \
                     'The rating scale is the same for all the questions and has five points:<br>' \
                     '"<i>Definitely not</i>", ' \
                     '"<i>Probably not</i>", ' \
                     '"<i>Unsure</i>", ' \
                     '"<i>Probably yes</i>", and ' \
                     '"<i>Definitely yes</i>".<br><br>' \
                     'The answer may seem more obvious for some combinations than for others. ' \
                     'Don\'t worry if you feel uncertain: go with your gut feeling and try your best ' \
                     'to capture your interpretation of the phrase.<br><br>' \
                     'It\'s also possible that the phrase may not make sense to you at all. ' \
                     'People\'s interpretations of nouns and adjectives (especially ones like "fake") can vary a lot! ' \
                     'If the phrase doesn\'t make sense to you, feel free to just use the "Unsure" option.<br><br>' \
                     'Let\'s start with some examples.'

    def __init__(self, bigram_id: str, bigram_path: str, mass_count_path: str):
        super().__init__(self.PAGE_BREAK_BETWEEN_BIGRAMS,
                         self.ONE_BLOCK_PER_GROUP_WITH_PAGE_BREAKS,
                         self.ONE_BIGRAM_PER_BLOCK,
                         self.SHUFFLE_BIGRAMS_WITHIN_BLOCKS,
                         bigram_id,
                         bigram_path,
                         self.TRAINING_BIGRAM_PATH,
                         self.FILLER_PATH,
                         self.SURVEY_OUT_DIR,
                         self.QUESTION_PHRASING,
                         self.MULTIPLE_CHOICE_ANSWERS,
                         self.TRAINING_INTRO,
                         fillers_from_template=self.FILLERS_FROM_TEMPLATE,
                         filler_question_phrasing=self.FILLER_QUESTION_PHRASING,
                         add_followup_question=False
                         )
        self.mass_count_labels = pd.read_csv(mass_count_path)

    def get_mass_count(self, noun: str) -> str:
        noun_row = self.mass_count_labels.loc[self.mass_count_labels['Noun'] == noun]
        if len(noun_row) == 0:
            raise ValueError(f'Noun {noun} not found in mass/count table')
        return noun_row['MassCount'].values[0]

    def format_question(self, bigram: Tuple[str, str], target_phrase: str = None, hint_text: str = None) -> str:
        """Generate the HTML for the question"""

        adjective, noun = bigram
        bigram_string = bigram_to_string(bigram)
        mass_count = self.get_mass_count(noun)
        bigram_det = get_determiner(bigram_string, mass_count)
        noun_det = get_determiner(noun, mass_count)
        verb = get_verb(mass_count)
        if target_phrase:
            full_question = build_filler_question(format_string=self.filler_question_phrasing,
                                                  bigram=bigram_string, bigram_det=bigram_det,
                                                  target_phrase=target_phrase, verb=verb)
        else:
            full_question = build_bigram_question(format_string=self.question_phrasing, bigram=bigram_string, noun=noun,
                                                  bigram_det=bigram_det, noun_det=noun_det, verb=verb)
        if hint_text:
            # This is a training question, add a hint/instructions for what the answer should be
            return f'{full_question}<br><br><em>{hint_text}</em>\n'
        else:
            return f'{full_question}\n'

    def format_followup_question(self, hint_text: str = None) -> str:
        raise NotImplementedError()


if __name__ == '__main__':
    # bigram_id = 'isa_part2'
    # bigram_path = '../../../bigrams/exp2_part2_latin_square.json'
    bigram_id = 'isa'
    bigram_path = '../../../bigrams/exp2_part1_latin_square.json'
    mass_count_csv_path = '../../../bigrams/nouns_mass_count.csv'
    generator = IsaSurveyGenerator(bigram_id=bigram_id, bigram_path=bigram_path, mass_count_path=mass_count_csv_path)
    generator.generate_qualtrics_questions()
