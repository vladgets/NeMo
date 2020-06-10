# =============================================================================
# Copyright 2020 NVIDIA. All Rights Reserved.
# Copyright 2018 The Google AI Language Team Authors and
# The HuggingFace Inc. team.
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

import numpy as np
from torch.utils.data import Dataset


class BartSummarizationDataset(Dataset):
    """
    Creates dataset to use for the task of summarization
    with pretrained Bart model.

    Converts from raw data to an instance that can be used by
    NMDataLayer.

    Args:
        queries (list): list of queries to run inference on
        max_seq_length (int): max sequence length
        tokenizer (Tokenizer): such as BartTokenizer
    """

    def __init__(self, queries, max_seq_length, tokenizer):
        dct = tokenizer.batch_encode_plus(
            queries, max_length=max_seq_length, return_tensors="pt", pad_to_max_length=True
        )

        self.all_input_ids = dct["input_ids"]
        self.all_input_mask = dct["attention_mask"]

    def __len__(self):
        return len(self.all_input_ids)

    def __getitem__(self, idx):
        return (np.array(self.all_input_ids[idx]), np.array(self.all_input_mask[idx], dtype=np.long))
