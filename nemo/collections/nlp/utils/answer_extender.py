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

# rules
replacement_rules = {"what's": "what is", "who's": "who is", "?": ""}
starting_phrases = ["tell me", "can you", "please"]
relation_words = ["from"]
question_words = ["what", "what about", "why", "who", "how old", "how long", "how", "do", "does", "where"]
verb_words = ["is", "are", "do", "does"]
belonging_word_convertor = {"my": "your", "your": "my", "yours": "my", "you": "i", "i" : "you",
                            "can you": "i can", "could you": "i could", "have you": "i have"}


# Function that extends the precise answer based on the question
def extend_answer(question, answer):
    print(question + " (" + answer + ") => ", end = " ")

    # Heuristics for long factual answer, where there is no need to copy question into the answer
    if len(answer.split()) >= 5:
        return answer[0].upper() + answer[1:]

    # move to lower case
    question = question.lower()

    # extend shortcuts in the question word and remove question sign
    for key in replacement_rules:
        question = question.replace(key, replacement_rules[key])

    # find possible relation word, like "from"
    relation_word = None
    for word in relation_words:
        if question.startswith(word):
            relation_word = word
            question = question[len(word) + 1:]
            break

    # find starting question word
    question_word = None
    for word in question_words:
        if question.startswith(word):
            question_word = word
            question = question[len(word)+1:]
            break

    # Try to find possible relation word like "from" for a second fime in case they are coming after the question
    if relation_word is None:
        for word in relation_words:
            if question.startswith(word):
                relation_word = word
                question = question[len(word) + 1:]
                break

    # in this case you do not add question to the answer
    if question_word is None:
        return answer

    # find starting verb word
    verb_word = None
    for word in verb_words:
        if question.startswith(word):
            if word is not "do" and word is not "does":
                if word == "are":
                    verb_word = "am"
                else:
                    verb_word = word
            question = question[len(word) + 1:]
            break

    # check if a phrase start with a belonging word that should be converted to the opposite one
    for key in belonging_word_convertor:
        if question.startswith(key):
            question = question.replace(key, belonging_word_convertor[key])
            break

    # create full answer
    full_answer = question

    if relation_word is not None:
        full_answer += " " + relation_word

    if verb_word is not None:
        full_answer += " " + verb_word

    full_answer += " " + answer

    # Capitalize the first letter
    full_answer = full_answer[0].upper() + full_answer[1:]

    return full_answer


# Uni-test of the module
if __name__ == "__main__":
    resp = extend_answer("what's your name", "Vlad")
    assert resp == "My name is Vlad", resp
    print(resp)

    resp = extend_answer("what's your favorite type of food", "mango")
    assert resp == "My favorite type of food is mango", resp
    print(resp)

    resp = extend_answer("what do you like to eat", "mango")
    assert resp == "I like to eat mango", resp
    print(resp)

    resp = extend_answer("how old are you", "22 years old")
    assert resp == "I am 22 years old", resp
    print(resp)

    resp = extend_answer("where do you live", "in the cloud")
    assert resp == "I live in the cloud", resp
    print(resp)

    resp = extend_answer("who is your favorite tennis player?", "novak djocovic")
    assert resp == "My favorite tennis player is novak djocovic", resp
    print(resp)

    resp = extend_answer("what can you talk about?", "the weather")
    assert resp == "I can talk about the weather", resp
    print(resp)

    resp = extend_answer("From where do you get the weather data?", "Weatherstack api")
    assert resp == "I get the weather data from Weatherstack api", resp
    print(resp)

    resp = extend_answer("Where from do you get the weather data?", "Weatherstack api")
    assert resp == "I get the weather data from Weatherstack api", resp
    print(resp)

    resp = extend_answer("What weather conditions cause fog?", "cold air passing over warmer water or moist land")
    assert resp == "Cold air passing over warmer water or moist land", resp
    print(resp)

    resp = extend_answer("How long have you studied the weather?", "all my life")
    assert resp == "I have studied the weather all my life", resp
    print(resp)

    resp = extend_answer("What can you do?", "I can talk about the weather")
    assert resp == "I can talk about the weather", resp
    print(resp)

    resp = extend_answer("How hail is formed?", "Hailstones are formed by layers of water attaching and freezing in a large cloud.")
    assert resp == "Hailstones are formed by layers of water attaching and freezing in a large cloud.", resp
    print(resp)