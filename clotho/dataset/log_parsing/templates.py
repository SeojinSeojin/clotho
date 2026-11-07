def messages_template(input_text):
    # input_text: the log message content (task2input_key['log_parsing'] == 'content')
    return [
        {"role": "system", "content": f"""You are an expert log parser."""},
        {"role": "user", "content": f"""You will be given one log message. Replace every variable part (IDs, numbers, paths, IPs) with <*> and keep the constant text unchanged.

Log message: {input_text}

Your answer must follow **exactly** this format:
Reasoning: <one short sentence on which tokens are variables>
Template: <the log template>"""}]


template_map = {
    "messages_template": messages_template,
}
