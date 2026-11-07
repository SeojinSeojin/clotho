def messages_template(input_text):
    # input_text: the log message content (task2input_key['log_parsing'] == 'content')
    return [
        {"role": "system", "content": f"""You are an expert log parser."""},
        {"role": "user", "content": f"""You will be given one log message. Replace every variable part (IDs, numbers, paths, IPs) with <*> and keep the constant text unchanged.

Log message: {input_text}

Your answer must follow **exactly** this format:
Reasoning: <one short sentence on which tokens are variables>
Template: <the log template>"""}]


def rules_template(input_text):
    # Same task and answer format as messages_template, plus the Loghub-2.0
    # labelling conventions (a whole variable token, including attached prefixes
    # and punctuation, becomes ONE <*>). Examples are generic, not taken from Loghub.
    return [
        {"role": "system", "content": f"""You are an expert log parser."""},
        {"role": "user", "content": f"""You will be given one log message. Replace every variable part with <*> and keep the constant text unchanged.

Rules:
1. A variable is a token whose value changes between occurrences of the same event: IDs, numbers, sizes, durations, IP addresses, ports, host names, file paths, URLs, user names.
2. Replace the WHOLE variable token with a single <*>, including any prefix, slash, colon or port attached to it. For example: req_8812_03 -> <*>, /192.168.0.7:8080 -> <*>, /var/log/app/run.log -> <*>, worker-3 -> <*>.
3. Keep fixed words, class names, exception names, and the punctuation and spacing between tokens exactly as they appear.
4. Do not add, remove, or reword anything else.

Log message: {input_text}

Your answer must follow **exactly** this format:
Reasoning: <one short sentence on which tokens are variables>
Template: <the log template>"""}]


template_map = {
    "messages_template": messages_template,
    "rules_template": rules_template,
}
