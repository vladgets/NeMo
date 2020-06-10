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

import argparse
import json

import nemo
from nemo.collections.nlp.nm.data_layers import BartSumarizationDataLayer
from nemo.collections.nlp.nm.trainables.common.huggingface.bart_nm import BartForConditionalGeneration


def generate_summaries(examples, model_name, batch_size=8):
    nf = nemo.core.NeuralModuleFactory(backend=nemo.core.Backend.PyTorch)

    tokenizer = nemo.collections.nlp.data.tokenizers.get_tokenizer(
        tokenizer_name='nemobert',
        # pretrained_model_name='facebook/bart-large'
        pretrained_model_name=model_name,
    )

    pretrained_bart_model = BartForConditionalGeneration(pretrained_model_name=model_name)

    data_layer = BartSumarizationDataLayer(queries=examples, tokenizer=tokenizer, batch_size=batch_size)
    input_data = data_layer()

    result = pretrained_bart_model(input_ids=input_data.input_ids, attention_mask=input_data.input_mask)

    summaries = nf.infer(tensors=[result],)

    cpu_summaries = summaries[0][0].numpy()
    print(cpu_summaries)
    result = [
        tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries[0][0]
    ]
    print(result)


def read_spotify_file(f, max_examples):
    examples = []
    for cnt, line in enumerate(f):
        if cnt >= max_examples:
            break
        json_data = json.loads(line)
        examples.append(json_data['text'])
    return examples


def run_generate():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source_file", type=str, help="like cnn_dm/test.source",
    )
    parser.add_argument(
        "output_file", type=str, help="where to save summaries",
    )
    parser.add_argument(
        "--spotify_format", action="store_true", help="if input file is stored in Spotify json format",
    )
    parser.add_argument(
        "--one_liners", action="store_true", help="if multiple one line examples in a single file",
    )
    parser.add_argument(
        "--model_name", type=str, default="facebook/bart-large-cnn", help="like facebook/bart-large-cnn",
    )
    parser.add_argument(
        "--bs", type=int, default=8, required=False, help="batch size: how many to summarize at a time",
    )
    args = parser.parse_args()

    f = open(args.source_file, 'r')
    # read input file (supporting different formats)
    if not args.spotify_format:
        if args.one_liners:
            examples = [" " + x.rstrip() for x in f.readlines()]
        else:
            example = ''
            for x in f.readlines():
                example += " " + x.rstrip()
            examples = [example]
    else:
        examples = read_spotify_file(f, 1)
    f.close()

    generate_summaries(examples, args.model_name, batch_size=args.bs)


if __name__ == "__main__":
    run_generate()
