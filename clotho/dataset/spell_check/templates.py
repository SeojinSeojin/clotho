def messages_template(input_text):
    return [
        {"role": "system", "content": f"""You are a helpful assistant tasked with detecting a misspelled word in the given sentence."""},
        {"role": "user", "content": f"""Find a misspelled word in the following sentence:

{input_text}

You are tasked to find only one misspelled word in the sentence. Pinpoint the most likely misspelled word.

Your answer must follow **exactly** this format:
Reasoning: <reason for the found misspelled word (brief as in one line)>
Misspelled word: <misspelled_word>
Corrected word: <corrected_word>

Do not include any other explanation or extra output."""}]
    

template_map = {
    "messages_template": messages_template,
}