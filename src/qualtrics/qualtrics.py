from random import shuffle
from typing import List, Tuple, Dict, Union

import pandas as pd
from utils.bigrams import string_to_bigram
from utils.io import read_json, read_bigrams, write_text_file


class QualtricsSurveyGenerator:
    def __init__(self, page_break_between_bigrams: bool,
                 one_block_per_group_with_page_breaks: bool,
                 one_bigram_per_block: bool,
                 shuffle_bigrams_within_blocks: bool,
                 bigram_id: str,
                 bigram_path: str,
                 training_bigram_path: str,
                 filler_bigram_path: str,
                 survey_out_dir: str,
                 question_phrasing: str,
                 multiple_choice_answers: List[str],
                 training_intro: str,
                 fillers_from_template: bool = True,
                 filler_question_phrasing: str = '',
                 add_followup_question: bool = False,
                 followup_question_phrasing: str = '',
                 followup_multiple_choice_answers: List[str] = [],

                 ):
        self.page_break_between_bigrams = page_break_between_bigrams
        self.one_block_per_group_with_page_breaks = one_block_per_group_with_page_breaks
        self.one_bigram_per_block = one_bigram_per_block
        self.add_followup_question = add_followup_question
        self.shuffle_bigrams_within_blocks = shuffle_bigrams_within_blocks

        self.bigram_latin_square_path = bigram_path
        self.training_bigram_path = training_bigram_path
        self.filler_bigram_path = filler_bigram_path
        self.fillers_from_template = fillers_from_template

        self.survey_out_path = f'{survey_out_dir}/survey_{bigram_id}' \
                               f'{"_block-group" if self.one_block_per_group_with_page_breaks else "_bigram-block" if self.one_bigram_per_block else ""}' \
                               f'{"_shuffled" if self.shuffle_bigrams_within_blocks else ""}' \
                               f'{"_followup" if self.add_followup_question else ""}.txt'

        self.question_phrasing = question_phrasing
        self.filler_question_phrasing = filler_question_phrasing
        self.question_type = 'Question:MC:SingleAnswer:Horizontal'
        self.multiple_choice_answers: List[str] = multiple_choice_answers
        self.followup_question_phrasing = followup_question_phrasing
        self.followup_question_type = 'Question:MC:SingleAnswer:Horizontal'
        self.followup_multiple_choice_answers: List[str] = followup_multiple_choice_answers

        self.training_intro = training_intro

    def format_question(self, bigram: Tuple[str, str], target_phrase: str = None, hint_text: str = None) -> str:
        raise NotImplementedError()

    def format_followup_question(self, hint_text: str = None) -> str:
        raise NotImplementedError()

    def generate_text_file_text(self, grouped_block_bigrams: List[List[List[Tuple[str, str]]]],
                                training_bigrams: List[Dict[str, str]] = None,
                                filler_bigrams: List[Tuple[str, str]] = None,
                                filler_questions: List[Tuple[Tuple[str, str], str]] = None) -> str:
        """

        :param grouped_block_bigrams: List of groups of blocks of bigrams:
               a list of groups, each group is a list of blocks, each block is a list of bigrams
               Typically, blocks_per_group = 1
        :param training_bigrams: List of bigram-hint tuples (bigram is a string-string tuple)
        :param filler_bigrams: List of bigrams as tuples to include in every block
            (works best with shuffling & one block per group, otherwise a group may see them more than once)
        :param filler_questions: List of questions to include verbatim in every block (see advice for filler bigrams)
        :return: text to write to Qualtrics survey text file
        """
        file_text = '[[AdvancedFormat]]\n\n'

        if training_bigrams:
            file_text += f'[[Block:TrainingIntro]]\n\n'
            file_text += self.generate_training_intro_question()

            # Always do training as one block with questions and page breaks, regardless of group / Latin square settings
            file_text += f'[[Block:Training]]\n\n'
            for i, bigram in enumerate(training_bigrams):
                target_phrase = bigram.get('target', None)
                file_text += self.generate_question((bigram['adjective'], bigram['noun']), hint_text=bigram['hint'],
                                                    target_phrase=target_phrase)
                if self.add_followup_question:
                    file_text += self.generate_followup_question((bigram['adjective'], bigram['noun']),
                                                                 bigram['followup_hint'])
                file_text += '[[PageBreak]]\n\n'

        if self.one_block_per_group_with_page_breaks:
            for i, group_bigrams in enumerate(grouped_block_bigrams):
                block_name = f'Block{i + 1}'
                file_text += f'[[Block:{block_name}]]\n\n'
                for block_bigrams in group_bigrams:
                    # If not shuffling, only include attention checks in first block so that they can be copied
                    # within Qualtrics (so that all respondents see literally the same attention checks, not duplicates)
                    filler_bigrams = filler_bigrams if i == 1 or self.shuffle_bigrams_within_blocks else []
                    filler_questions = filler_questions if i == 1 or self.shuffle_bigrams_within_blocks else []
                    file_text += self.generate_block_questions(block_bigrams, filler_bigrams, filler_questions)

                    if not self.page_break_between_bigrams:
                        # No point in adding a page break between blocks if we already have a page break after questions
                        file_text += '[[PageBreak]]\n\n'

        else:
            bigrams_by_blocks = self.flatten_groups(grouped_block_bigrams)
            for i, block_bigrams in enumerate(bigrams_by_blocks):
                # See comment above
                filler_bigrams = filler_bigrams if i == 1 or self.shuffle_bigrams_within_blocks else []
                file_text += self.generate_block_questions(block_bigrams, filler_bigrams, block_id=i + 1)

        return file_text

    def generate_block_questions(self, block_bigrams: List[Tuple[str, str]],
                                 filler_bigrams: List[Tuple[str, str]] = None,
                                 filler_questions: List[Tuple[Tuple[str, str], str]] = None,
                                 block_id: int = None) -> str:
        block_text = ""
        if block_id and not self.one_bigram_per_block:
            block_name = f'Block{block_id}'
            block_text += f'[[Block:{block_name}]]\n\n'

        these_bigrams: List[Union[Tuple[str, str], Tuple[Tuple[str, str], str]]] = block_bigrams
        if filler_bigrams:
            these_bigrams.extend(filler_bigrams)
        if filler_questions:
            these_bigrams.extend(filler_questions)
        if self.shuffle_bigrams_within_blocks:
            shuffle(these_bigrams)

        for j, bigram_or_filler in enumerate(these_bigrams):
            block_text += self.generate_block_question(block_id, j, bigram_or_filler)

        return block_text

    def generate_block_question(self, block_id: int, question_number: int,
                                bigram_or_filler: Union[Tuple[str, str], Tuple[Tuple[str, str], str]]):
        if isinstance(bigram_or_filler[0], str):
            bigram = bigram_or_filler
            target = None
        else:
            bigram = bigram_or_filler[0]
            target = bigram_or_filler[1]

        block_text = ''
        if self.one_bigram_per_block:
            block_name = f'Block{block_id}Q{question_number + 1}'
            block_text += f'[[Block:{block_name}]]\n\n'
        block_text += self.generate_question(bigram, target)
        if self.add_followup_question:
            block_text += self.generate_followup_question(bigram)
        if self.page_break_between_bigrams:
            block_text += '[[PageBreak]]\n\n'
        return block_text

    def generate_training_intro_question(self) -> str:
        question_text = f'[[Question:Text]]\n'
        question_text += f'[[ID:TrainingIntro]]\n'
        question_text += self.training_intro
        question_text += '\n\n'

        return question_text

    def generate_question(self, bigram: Tuple[str, str], target_phrase: str = None, hint_text: str = None) -> str:
        """
        n.B. this does not make the questions required - you have to do that manually in Qualtrics by
        selecting all the questions at once and setting them as required
        """
        question_text = f'[[{self.question_type}]]\n'
        question_text += f'[[ID:Q-{bigram[0]}-{bigram[1]}]]\n'
        question_text += self.format_question(bigram, target_phrase=target_phrase, hint_text=hint_text)

        question_text += '[[Choices]]\n'
        for choice in self.multiple_choice_answers:
            question_text += f'{choice}\n'
        question_text += '\n'

        return question_text

    def generate_followup_question(self, bigram: Tuple[str, str], hint_text: str = None) -> str:
        question_text = f'[[{self.followup_question_type}]]\n'
        question_text += f'[[ID:Q-{bigram[0]}-{bigram[1]}-FUP]]\n'
        question_text += self.format_followup_question(hint_text)

        question_text += '[[Choices]]\n'
        for choice in self.followup_multiple_choice_answers:
            question_text += f'{choice}\n'
        question_text += '\n'

        return question_text

    def flatten_groups(self, grouped_block_bigrams: List[List[List[Tuple[str, str]]]]) -> List[List[Tuple[str, str]]]:
        block_bigrams = []
        for group_bigrams in grouped_block_bigrams:
            block_bigrams.extend(group_bigrams)

        return block_bigrams

    def filler_question_df_to_tuples(self, filler_df: pd.DataFrame) -> List[Tuple[Tuple[str, str], str]]:
        filler_rows: List[List[str]] = filler_df.to_numpy().tolist()
        filler_tuples = [(string_to_bigram(row[0]), row[1]) for row in filler_rows]
        return filler_tuples

    def generate_qualtrics_questions(self):
        if self.one_bigram_per_block and self.page_break_between_bigrams:
            raise ValueError("No need for page break between bigrams if you're doing one bigram per block")
        if self.one_bigram_per_block and self.one_block_per_group_with_page_breaks:
            raise ValueError("You must pick exactly one of: one bigram per block, one block per group, or neither")

        survey_bigrams = read_json(self.bigram_latin_square_path)
        train_bigrams = read_json(self.training_bigram_path)

        if self.fillers_from_template:
            filler_bigrams = read_bigrams(self.filler_bigram_path)
            survey_file_text = self.generate_text_file_text(survey_bigrams, train_bigrams,
                                                            filler_bigrams=filler_bigrams)
        else:
            filler_df = pd.read_csv(self.filler_bigram_path)
            filler_questions = self.filler_question_df_to_tuples(filler_df)
            survey_file_text = self.generate_text_file_text(survey_bigrams, train_bigrams,
                                                            filler_questions=filler_questions)

        write_text_file(survey_file_text, out_path=self.survey_out_path)
