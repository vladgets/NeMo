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

from typing import List, Optional

from transformers import BART_PRETRAINED_MODEL_ARCHIVE_LIST, BartConfig
from transformers import BartForConditionalGeneration as HF_BartForConditionalGeneration
from transformers.configuration_bart import BART_PRETRAINED_CONFIG_ARCHIVE_MAP

from nemo.backends.pytorch.nm import TrainableNM
from nemo.core.neural_modules import PretrainedModelInfo
from nemo.core.neural_types import ChannelType, NeuralType
from nemo.utils.decorators import add_port_docs


class BartForConditionalGeneration(TrainableNM):
    """
        BartForConditionalGeneration wraps around the Huggingface implementation of BART model for summarization
        from their transformers repository for easy use within NeMo.
    """

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        input_ids: input token ids
        attention_mask: attention mask for paddings (0/1)
        """
        return {
            "input_ids": NeuralType(('B', 'T'), ChannelType()),
            "attention_mask": NeuralType(('B', 'T'), ChannelType()),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        hidden_states: output embedding
        """
        return {"hidden_states": NeuralType(('B', 'T', 'D'), ChannelType())}

    def __init__(self, pretrained_model_name=None, config_filename=None):
        """
        Args:
            pretrained_model_name (str): If using a pretrained model, this should
                be the model's name. Otherwise, should be left as None.
            config_filename (str): path to model configuration file. Optional.
        """
        super().__init__()

        if pretrained_model_name is not None:
            model = HF_BartForConditionalGeneration.from_pretrained(pretrained_model_name)
        elif config_filename is not None:
            config = BartConfig.from_json_file(config_filename)
            model = HF_BartForConditionalGeneration(config)
        else:
            raise ValueError(
                "Either pretrained_model_name or config_filename must be passed into the BERT constructor"
            )

        model.to(self._device)

        self.add_module("bart", model)
        self.config = model.config
        self.hidden_size = model.config.hidden_size

    def forward(self, input_ids, attention_mask):
        """
            2 different modes - training and generation in inference mode
        """
        if self.training:
            return self.bart(input_ids, attention_mask=attention_mask)[0]
        else:
            # Currently using parameters which optimized for generating summaries for CNN/Daily_mail data set
            # for other types of summaries, for example short ones like for X-Net data set values of optimized
            # generate parameters are different ones
            max_length = 140
            min_length = 55

            return self.bart.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=4,
                length_penalty=2.0,
                max_length=max_length + 2,  # +2 from original because we start at step=1 and stop before max_length
                min_length=min_length + 1,  # +1 from original because we start at step=1
                no_repeat_ngram_size=3,
                early_stopping=True,
                decoder_start_token_id=self.config.eos_token_id,
            )

    @staticmethod
    def list_pretrained_models() -> Optional[List[PretrainedModelInfo]]:
        pretrained_models = []
        for key, value in BART_PRETRAINED_MODEL_ARCHIVE_MAP.items():
            model_info = PretrainedModelInfo(
                pretrained_model_name=key,
                description="weights by HuggingFace",
                parameters=BART_PRETRAINED_CONFIG_ARCHIVE_MAP[key],
                location=value,
            )
            pretrained_models.append(model_info)
        return pretrained_models
