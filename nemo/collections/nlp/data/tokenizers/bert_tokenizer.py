# =============================================================================
# Copyright 2020 NVIDIA. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import re

from nemo import logging
from nemo.collections.nlp.data.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.nlp.data.tokenizers.tokenizer_utils import MODEL_SPECIAL_TOKENS, TOKENIZERS

__all__ = [
    'NemoBertTokenizer',
]


def handle_quotes(text):
    text_ = ""
    quote = 0
    i = 0
    while i < len(text):
        if text[i] == "\"":
            if quote % 2:
                text_ = text_[:-1] + "\""
            else:
                text_ += "\""
                i += 1
            quote += 1
        else:
            text_ += text[i]
        i += 1
    return text_


def remove_spaces(text):
    text = text.replace("( ", "(")
    text = text.replace(" )", ")")
    text = text.replace("[ ", "[")
    text = text.replace(" ]", "]")
    text = text.replace(" / ", "/")
    text = text.replace("„ ", "„")
    text = text.replace(" - ", "-")
    text = text.replace(" ' ", "'")
    text = re.sub(r'([0-9])( )([\.,])', '\\1\\3', text)
    text = re.sub(r'([\.,])( )([0-9])', '\\1\\3', text)
    text = re.sub(r'([0-9])(:)( )([0-9])', '\\1\\2\\4', text)
    text = text.replace(" %", "%")
    text = text.replace("$ ", "$")
    text = re.sub(r'([^0-9])(,)([0-9])', '\\1\\2 \\3', text)
    return text


class NemoBertTokenizer(TokenizerSpec):
    def __init__(self, pretrained_model=None, vocab_file=None, bert_derivative='bert', do_lower_case=False):
        '''
        The user needs to specify pretrained_model name or vocab file and bert_derivative

        pretrained_model (str):name of the pretrained model from the hugging face list,
            for example: bert-base-cased
            To see the list of pretrained models, call:
            huggingface_utils.get_bert_models_list()
        vocab_file: File containing the vocabulary.
        bert_derivative: for example: 'bert', 'roberta', 'albert'. Only used when vocab_file specified.
        '''
        if pretrained_model:
            bert_derivative = pretrained_model.split('-')[0]
            logging.info(f'Deriving bert model type from pretrained model name.')

        if bert_derivative in TOKENIZERS:
            tokenizer_cls = TOKENIZERS[bert_derivative]
        else:
            raise ValueError(
                "Bert_derivative value {bert_derivative} is not currently supported"
                + " Please choose from the following list: {TOKENIZERS.keys()}"
            )

        if pretrained_model:
            self.tokenizer = tokenizer_cls.from_pretrained(pretrained_model)
        elif vocab_file:
            self.tokenizer = tokenizer_cls(vocab_file=vocab_file, do_lower_case=do_lower_case)
        else:
            raise ValueError("either 'vocab_file' or 'pretrained_model' has to be specified")

        if hasattr(self.tokenizer, "vocab"):
            self.vocab_size = len(self.tokenizer.vocab)

        special_tokens = MODEL_SPECIAL_TOKENS[bert_derivative]
        for k, v in special_tokens.items():
            setattr(self, k, v)
        self.never_split = tuple(special_tokens.values())

    def text_to_tokens(self, text):
        tokens = self.tokenizer.tokenize(text)
        return tokens

    def tokens_to_text(self, tokens):
        text = self.tokenizer.convert_tokens_to_string(tokens)
        return remove_spaces(handle_quotes(text.strip()))

    def token_to_id(self, token):
        return self.tokens_to_ids([token])[0]

    def tokens_to_ids(self, tokens):
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return ids

    def ids_to_tokens(self, ids):
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        return tokens

    def text_to_ids(self, text):
        tokens = self.text_to_tokens(text)
        ids = self.tokens_to_ids(tokens)
        return ids

    def ids_to_text(self, ids):
        tokens = self.ids_to_tokens(ids)
        tokens_clean = [t for t in tokens if t not in self.never_split]
        text = self.tokens_to_text(tokens_clean)
        return text

    def batch_encode_plus(self, text, max_length=None, pad_to_max_length=False, return_tensors=None):
        """
        Wrapper to an appropriate function from HF.
        Returns a dictionary containing the encoded sequence or sequence pair and additional information:
        the mask for sequence classification and the overflowing elements if a ``max_length`` is specified.

        Args:
            text (:obj:`str`, :obj:`List[str]` or :obj:`List[int]` (the later only for not-fast tokenizers)):
            The first sequence to be encoded. This can be a string, a list of strings (tokenized string using
            the `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
            method)
            max_length (:obj:`int`, `optional`, defaults to :obj:`None`):
                If set to a number, will limit the total sequence returned so that it has a maximum length.
                If there are overflowing tokens, those will be added to the returned dictionary
                You can set it to the maximal input size of the model with `max_length = tokenizer.model_max_length`.
            pad_to_max_length (:obj:`bool`, `optional`, defaults to :obj:`False`):
                If set to True, the returned sequences will be padded according to the model's padding side and
                padding index, up to their max length. If no max length is specified, the padding is done up to the
                model's max length. The tokenizer padding sides are handled by the class attribute `padding_side`
                which can be set to the following strings:
                - 'left': pads on the left of the sequences
                - 'right': pads on the right of the sequences
                Defaults to False: no padding.
            return_tensors (:obj:`str`, `optional`, defaults to :obj:`None`):
                Can be set to 'tf' or 'pt' to return respectively TensorFlow :obj:`tf.constant`
                or PyTorch :obj:`torch.Tensor` instead of a list of python integers.
        """
        tokens = self.tokenizer.batch_encode_plus(
            text, max_length=max_length, pad_to_max_length=pad_to_max_length, return_tensors=return_tensors
        )
        return tokens

    def decode(self, text, skip_special_tokens=False, clean_up_tokenization_spaces=True):
        """
        Wrapper to an appropriate function from HF.
        Converts a sequence of ids (integer) in a string, using the tokenizer and vocabulary
        with options to remove special tokens and clean up tokenization spaces.
        Args:
            token_ids: list of tokenized input ids. Can be obtained using the `encode` or `encode_plus` methods.
            skip_special_tokens: if set to True, will replace special tokens.
            clean_up_tokenization_spaces: if set to True, will clean up the tokenization spaces.
        Returns:
            The decoded string
        """
        text = self.tokenizer.decode(text, skip_special_tokens, clean_up_tokenization_spaces)
        return text

    @property
    def pad_id(self):
        return self.tokens_to_ids([getattr(self, 'pad_token')])[0]

    @property
    def bos_id(self):
        return self.tokens_to_ids([getattr(self, 'bos_token')])[0]

    @property
    def eos_id(self):
        return self.tokens_to_ids([getattr(self, 'eos_token')])[0]

    @property
    def sep_id(self):
        return self.tokens_to_ids([getattr(self, 'sep_token')])[0]

    @property
    def cls_id(self):
        return self.tokens_to_ids([getattr(self, 'cls_token')])[0]
