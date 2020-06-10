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

from nemo.collections.nlp.data.datasets.summarization_bart_dataset import BartSummarizationDataset
from nemo.collections.nlp.nm.data_layers.text_datalayer import TextDataLayer
from nemo.core import ChannelType, LabelsType, NeuralType
from nemo.utils.decorators import add_port_docs

__all__ = ['BartSumarizationDataLayer']


class BartSumarizationDataLayer(TextDataLayer):
    """
    Creates the data layer to use for the task of summarization
    with pretrained Bart model.

    All the data processing is done BartSummarizationDataset.
    """

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        input_ids:
            indices of tokens which constitute batches of masked text segments
        input_mask:
            bool tensor with 0s in place of tokens to be masked
        labels: sequence classification id
        """
        return {
            "input_ids": NeuralType(('B', 'T'), ChannelType()),
            "input_mask": NeuralType(('B', 'T'), ChannelType()),
            # "labels": NeuralType(tuple('B'), LabelsType()),
        }

    def __init__(
        self,
        queries,
        tokenizer,
        max_seq_length=1024,
        shuffle=False,
        batch_size=8,
        dataset_type=BartSummarizationDataset,
    ):
        """
            Args:
                queries (list): list of queries to run inference on
                tokenizer (TokenizerSpec): text tokenizer.
                max_seq_length (int): max sequence length
                shuffle (bool): whether to shuffle data or not. Default: False.
                batch_size: text segments batch size
                dataset (BartSummarizationDataset): the dataset that needs to be converted to DataLayerNM
        """
        dataset_params = {
            'queries': queries,
            'tokenizer': tokenizer,
            'max_seq_length': max_seq_length,
        }
        super().__init__(dataset_type, dataset_params, batch_size, shuffle=shuffle)
