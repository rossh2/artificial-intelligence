from typing import List, Tuple

from qualtrics.qualtrics import QualtricsSurveyGenerator


class FilteringSurveyGenerator(QualtricsSurveyGenerator):
    PAGE_BREAK_BETWEEN_BIGRAMS = True
    ONE_BLOCK_PER_GROUP_WITH_PAGE_BREAKS = True
    ONE_BIGRAM_PER_BLOCK = False
    ADD_FOLLOWUP_QUESTION = False
    SHUFFLE_BIGRAMS_WITHIN_BLOCKS = True

    BIGRAM_ID = 'analogy60x12_bigrams'
    BIGRAM_PATH = f'../../../output/filtering_data/grouped_{BIGRAM_ID}.json'
    TRAINING_BIGRAM_PATH = '../../../bigrams/filtering_training_bigrams.json'
    ATTENTION_CHECK_BIGRAM_PATH = '../../../bigrams/attention_check_bigrams.txt'
    SURVEY_OUT_DIR = f'../surveys'

    QUESTION_PHRASING = 'How easy is it to imagine what this would mean?'
    QUESTION_TYPE = 'Question:MC:SingleAnswer:Horizontal'
    MULTIPLE_CHOICE_ANSWERS: List[str] = [
        'Very hard',
        'Somewhat hard',
        'Somewhat easy',
        'Very easy'
    ]
    FOLLOWUP_QUESTION_PHRASING = 'Is it easy to think of multiple very different meanings?'
    FOLLOWUP_QUESTION_TYPE = 'Question:MC:SingleAnswer:Horizontal'
    FOLLOWUP_MULTIPLE_CHOICE_ANSWERS: List[str] = [
        'No',
        # 'Somewhat',
        'Yes'
    ]

    assert len(MULTIPLE_CHOICE_ANSWERS) == 4  # The text "four" is used in the intro below
    # Also make sure that the question phrasing matches the actual question phrasing
    TRAINING_INTRO = 'Thank you for taking part!' \
                     '<br><br>' \
                     'On each page, you will be given an adjective-noun pair like ' \
                     '<strong>"blue rainbow"</strong>. ' \
                     'You will then be asked how easy it is to imagine what that might mean. ' \
                     'The rating scale has four points: <br>' \
                     f'<em>\'{MULTIPLE_CHOICE_ANSWERS[0]}\'</em>, ' \
                     f'<em>\'{MULTIPLE_CHOICE_ANSWERS[1]}\'</em>, ' \
                     f'<em>\'{MULTIPLE_CHOICE_ANSWERS[2]}\'</em>, and ' \
                     f'<em>\'{MULTIPLE_CHOICE_ANSWERS[3]}\'</em>. ' \
                     '<br>'
    if ADD_FOLLOWUP_QUESTION:
        TRAINING_INTRO += 'A second question will then ask you whether it\'s actually ' \
                          'easy to think of more than one meaning.' \
                          '<br>'
    TRAINING_INTRO += '<br>' \
                      'Don\'t worry if you feel uncertain: ' \
                      'go with your gut feeling of whether you can imagine the phrase or not. ' \
                      '<br><br>' \
                      'Let\'s start with some examples.'

    def __init__(self):
        super().__init__(self.PAGE_BREAK_BETWEEN_BIGRAMS,
                         self.ONE_BLOCK_PER_GROUP_WITH_PAGE_BREAKS,
                         self.ONE_BIGRAM_PER_BLOCK,
                         self.SHUFFLE_BIGRAMS_WITHIN_BLOCKS,
                         self.BIGRAM_ID,
                         self.BIGRAM_PATH,
                         self.TRAINING_BIGRAM_PATH,
                         self.ATTENTION_CHECK_BIGRAM_PATH,
                         self.SURVEY_OUT_DIR,
                         self.QUESTION_PHRASING,
                         self.MULTIPLE_CHOICE_ANSWERS,
                         self.TRAINING_INTRO,
                         add_followup_question=self.ADD_FOLLOWUP_QUESTION,
                         followup_question_phrasing=self.FOLLOWUP_QUESTION_PHRASING,
                         followup_multiple_choice_answers=self.FOLLOWUP_MULTIPLE_CHOICE_ANSWERS)

    def format_question(self, bigram: Tuple[str, str], target_phrase: str = None, hint_text: str = None) -> str:
        """Generate the HTML for the question"""

        adjective, noun = bigram
        if hint_text:
            # This is a training question, add a hint/instructions for what the answer should be
            return f'<em>{hint_text}</em><br><br><strong>{adjective} {noun}</strong><br><br>{self.question_phrasing}\n'
        else:
            return f'<strong>{adjective} {noun}</strong><br><br>{self.question_phrasing}\n'

    def format_followup_question(self, hint_text: str = None) -> str:
        if hint_text:
            return f'<em>{hint_text}</em><br><br>{self.followup_question_phrasing}\n'
        else:
            return f'{self.followup_question_phrasing}\n'


if __name__ == '__main__':
    generator = FilteringSurveyGenerator()
    generator.generate_qualtrics_questions()
