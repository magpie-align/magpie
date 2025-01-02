import re

def input_difficulty_rating(input):
    user_message = f'''
# Instruction 

You first need to identify the given user intent and then label the difficulty level of the user query based on the content of the user query.

## User Query
```
{input}
```

## Output Format
Given the user query, in your output, you first need to identify the user intent and the knowledge needed to solve the task in the user query.
Then, rate the difficulty level of the user query as `very easy`, `easy`, `medium`, `hard`, or `very hard`.

Now, please output the user intent and difficulty level below in a json format by filling in the placeholders in []:
```
{{   
    "intent": "The user wants to [....]",
    "knowledge": "To solve this problem, the models need to know [....]",
    "difficulty": "[very easy/easy/medium/hard/very hard]"
}}
```
'''
    return user_message

def input_classification(input):
    user_message = f'''
# Instruction

Please label the task tags for the user query.

## User Query
```
{input}
```

## Tagging the user input
Please label the task tags for the user query. You will need to analyze the user query and select the most relevant task tag from the list below.

all_task_tags = [
    "Information seeking",  # Users ask for specific information or facts about various topics.
    "Reasoning",  # Queries require logical thinking, problem-solving, or processing of complex ideas.
    "Planning",  # Users need assistance in creating plans or strategies for activities and projects.
    "Editing",  # Involves editing, rephrasing, proofreading, or other tasks related to the composition of general written content.
    "Coding & Debugging",  # Users seek help with writing, reviewing, or fixing code in programming.
    "Math",  # Queries related to mathematical concepts, problems, and calculations.
    "Role playing",  # Users engage in scenarios requiring ChatGPT to adopt a character or persona.
    "Data analysis",  # Requests involve interpreting data, statistics, or performing analytical tasks.
    "Creative writing",  # Users seek assistance with crafting stories, poems, or other creative texts. 
    "Advice seeking",  # Users ask for recommendations or guidance on various personal or professional issues.
    "Brainstorming",  # Involves generating ideas, creative thinking, or exploring possibilities. 
    "Others"  # Any queries that do not fit into the above categories or are of a miscellaneous nature.
]

## Output Format:
Note that you can only select a single primary tag. Other applicable tags can be added to the list of other tags.
Now, please output your tags below in a json format by filling in the placeholders in <...>:
```
{{ 
    "primary_tag": "<primary tag>",
    "other_tags": ["<tag 1>", "<tag 2>", ... ]
}}
```
'''
    return user_message

def input_quality_rating(input):
    user_message = f'''
# Instruction

You need to rate the quality of the user query based on its clarity, specificity, and coherence.

The rating scale is as follows:

- very poor: The query is unclear, vague, or incoherent. It lacks essential information and context.
- poor: The query is somewhat unclear or lacks important details. It requires significant clarification.
- average: The query is moderately clear and specific. It may require some additional information for a complete understanding.
- good: The query is clear, specific, and mostly well-formed. It provides sufficient context for understanding the user's intent.
- excellent: The query is very clear, specific, and well-articulated. It contains all the necessary information and context for providing a comprehensive response.

## User Query
```
{input}
```

## Output Format
Given the user query, you first need to give an assesement, highlighting the strengths and/or weaknesses of the user query.
Then, you need to output a rating from very poor to excellent by filling in the placeholders in [...]:
```
{{   
    "explanation": "[...]",
    "input_quality": "[very poor/poor/average/good/excellent]"
}}
```
'''
    return user_message


# For statistcal analysis
MATH_KEYWORDS = [
    "radius", "diameter", "circumference", "perimeter", "triangle", "rectangle",
    "algebra", "geometry", "calculus", "trigonometry", "set theory", "linear regression",
    "number theory", "graph theory", "topology", "differential equations", "integral", "derivative",
    "logarithm", "exponent", "polynomial", "quadratic", "vector", "matrix", "determinant", "eigenvalue",
    "eigenvector", "complex number", "real number", "rational number", "irrational number", "prime number",
    "factorial", "permutation", "combination", "binomial", "continuity", "eigen vector",
    "domain", "range", "inverse", "composition", "angle", "radian", "sine",
    "cosine", "tangent", "cotangent", "secant", "cosecant", "pythagorean theorem", "euclidean geometry",
    "non-euclidean geometry", "hyperbolic geometry", "elliptic geometry", "fractal", "chaos theory",
    "golden ratio", "fibonacci sequence", "pascal's triangle", "binomial theorem", "fourier series",
    "laplace transform", "z-transform", "normal distribution", "poisson distribution", "chi-square distribution",
    "Student's t-distribution", "confidence interval", "hypothesis testing", "p-value", "bayes' theorem",
    "law of large numbers", "central limit theorem", "regression analysis", "correlation coefficient",
    "covariance", "ANOVA", "chi-square test", "t-test", "F-test", "Mann-Whitney U test", "Kolmogorov-Smirnov test",
    "Wilcoxon signed-rank test", "Kruskal-Wallis test", "Fisher's exact test", "McNemar's test", "Spearman's rank correlation",
    "Kendall's tau", "Pearson's correlation", "linear programming", "integer programming", "dynamic programming",
]

CODE_KEYWORDS = [
    "Python", "Java", "in C", "C++", "C#", "JavaScript", "Ruby", "PHP", "Swift", "Objective-C",
    "Kotlin", "in Go", "Rust", "Lua", "Perl", "MATLAB", "Fortran", "Julia", "TypeScript", 
    "F#", "Bash", "PowerShell", "SQL", "HTML", "CSS", "XML", "YAML", "JSON", 
    "Markdown", "LaTeX", "Assembly", "VBA", "Pascal",
    "in D", "in R", "Solidity", "Verilog", "VHDL", "Visual Basic", "BASIC",
    "\n\n```", "\n```", "```", "import ", "def", "elif", "#include"
]


def find_mcq_end(string):
    patterns = [
        r'\([A-D]\)',
        r'[A-D][\.\)]',
        r'[a-d][\.\)]',
    ]
    
    combined_pattern = '|'.join(patterns)
    matches = list(re.finditer(combined_pattern, string, re.IGNORECASE))
    
    if len(matches) >= 4:
        last_match = matches[-1]
        return True, last_match.end()
    
    return False, -1

def find_next_newline(string):
    single_newline = string.find('\n')
    double_newline = string.find('\n\n')
    
    if single_newline == -1 and double_newline == -1:
        return -1
    elif single_newline == -1:
        return double_newline
    elif double_newline == -1:
        return single_newline
    else:
        return min(single_newline, double_newline)

def generate_variants(prefixes):
    variants = []
    for word in prefixes:
        clean_word = word.lstrip('#* ').rstrip(':')
        variants.append(f"{clean_word}:")
        variants.append(f"**{clean_word}**")
        variants.append(f"## {clean_word}:")
    
    return variants

def remove_prefix(string):
    # Remove numbers and alphabets with periods or brackets at the beginning
    string = re.sub(r'^(\d+\.|\w+\))\s*', '', string).strip()
    
    # Remove prefixes like "Task:", "Prompt:", "Question:"
    prefixes = [
    "Task", "Prompt", "Original Prompt", "Question", "The Problem", "Problem",
    "Scenario", "The senario", "Situation", "The situation", "Context",
    "Challenge", "Query", "Request", "Instructions", "Instruction",
    "Descriptions", "Description"
    ]
    prefixes = generate_variants(prefixes)

    prefix_patterns = [re.escape(prefix) for prefix in prefixes]
    
    prefix_patterns.append(r"Question \d+:") # Question 1:, Question 2:, etc.
    prefix_patterns.append(r"Question \d+\.") # Question 1., Question 2., etc.
    prefix_patterns.append(r"Part \d+:") # Part 1:, Part 2:, etc.
    prefix_patterns.append(r"Q\d+\.") # Q1., Q2., etc.
    prefix_patterns.append(r"Q\d+:") # Q1:, Q2:, etc.
    
    combined_pattern = "|".join(prefix_patterns)
    
    match = re.match(f"^({combined_pattern})\s*", string, re.IGNORECASE)
    if match:
        return string[match.end():].strip()
    
    return string


def instruction_post_process(instruction, model_path):
    if "gemma-2" in model_path.lower():
        # remove the prefix
        instruction = remove_prefix(instruction)
        # find mcq problems
        is_mcq, end_pos = find_mcq_end(instruction)
        assistant_markers = ["Answer:", "Answers:", "The answer is", "Correct answer", "The correct answer", "Answer is", "Explanation:", "Here are some", "Solution Approach:", "Solution:"]
        assistant_pattern = r'(?:' + '|'.join(assistant_markers) + ')s?:'
        assistant_match = re.search(assistant_pattern, instruction) # Exact match, no re.IGNORECASE

        # TODO
        # if is_mcq:
        #     print(f"MCQ detected: {instruction}")
        #     rest_of_string = instruction[end_pos:]
        #     newline_pos = find_next_newline(rest_of_string)
        #     if newline_pos != -1:
        #         instruction = instruction[:end_pos + newline_pos].strip()
        #     else:
        #         instruction = instruction.strip()
        #     print(f"Sanitized MCQ instruction: {instruction}")
        #     class_num = 0

        if instruction.startswith("*"):
            if '?' in instruction:
                instruction = instruction.split('?')[0].replace("*", "").strip() + '?'
                instruction = remove_prefix(instruction)
                class_num = 1
                return instruction, class_num

        instruction = remove_prefix(instruction)

        if instruction.startswith('\"'):
            if '?' in instruction:
                instruction = instruction.split('?')[0].replace("\"", "").strip() + '?'
                class_num = 2.1
            else:
                instruction = instruction.split('\n')[0].replace("\"", "").strip()
                instruction = instruction.replace("*", "").strip()  
                class_num = 2.2
        elif instruction.startswith('<b>'):
            instruction.split('\n')[0].replace('</b>', "").replace('<b>', "").strip()
            instruction = instruction.replace("*", "").strip()
            class_num = 3
        elif assistant_match:
            instruction = instruction[:assistant_match.start()].strip()
            instruction = instruction.replace("**", "").strip() if instruction.find("**") == 1 else instruction.strip()
            class_num = 4
        elif instruction.split('\n')[0].strip().endswith(':'):
            colon_pos = instruction.split('\n')[0].strip().rfind(':')
            if '#' in instruction:
                instruction = instruction.split('#')[0].strip()
                instruction = instruction.replace("**", "").strip() if instruction.find("**") == 1 else instruction.strip()
                class_num = 5.1
            elif '?' in instruction:
                instruction = instruction.split('?')[0].strip() + '?'
                instruction = instruction.replace("**", "").strip() if instruction.find("**") == 1 else instruction.strip()
                class_num = 5.2
            else:
                instruction = instruction.split('\n')[0].strip()
                instruction = instruction.replace("*", "").strip()
                class_num = 5.3
        else:
            if '?' in instruction:
                instruction = instruction.split('?')[0].strip() + '?'
                instruction = instruction.replace("**", "").strip() if instruction.find("**") == 1 else instruction.strip()
                class_num = 6.1
            else:
                instruction = instruction.split('\n')[0].strip()
                instruction = instruction.replace("*", "").strip()
                class_num = 6.2

        # Remove prefixes again
        instruction = remove_prefix(instruction)

        return instruction, class_num

    elif "llama-3" in model_path.lower():
        # remove the prefix
        instruction = remove_prefix(instruction)
        # find mcq problems
        is_mcq, end_pos = find_mcq_end(instruction)
        assistant_markers = ["Answer:", "Answers:", "The answer is", "Correct answer", "The correct answer", "Answer is", "Explanation:", "Here are some", "Solution Approach:", "Solution:"]
        assistant_pattern = r'(?:' + '|'.join(assistant_markers) + ')s?:'
        assistant_match = re.search(assistant_pattern, instruction) # Exact match, no re.IGNORECASE
        # print(f"Assistant match: {assistant_match}")

        step_makers = ["# Step 1", "## Step 1", "### Step 1"]
        step_pattern = r'(?:' + '|'.join(step_makers) + r'):?'
        step_match = re.search(step_pattern, instruction) # Exact match, no re.IGNORECASE
        # print(f"Step match: {step_match}")

        # TODO
        # if is_mcq:
        #     print(f"MCQ detected: {instruction}")
        #     rest_of_string = instruction[end_pos:]
        #     newline_pos = find_next_newline(rest_of_string)
        #     if newline_pos != -1:
        #         instruction = instruction[:end_pos + newline_pos].strip()
        #     else:
        #         instruction = instruction.strip()
        #     print(f"Sanitized MCQ instruction: {instruction}")
        #     class_num = 0

        if instruction.startswith("*"):
            if '?' in instruction:
                instruction = instruction.split('?')[0].replace("*", "").strip() + '?'
                instruction = remove_prefix(instruction)
                class_num = 1
                return instruction, class_num

        instruction = remove_prefix(instruction)
        if instruction.startswith('\"'):
            if '?' in instruction:
                instruction = instruction.split('?')[0].replace("\"", "").strip() + '?'
                class_num = 2.1
            else:
                instruction = instruction.split('\n')[0].replace("\"", "").strip()
                instruction = instruction.replace("*", "").strip()  
                class_num = 2.2
        elif instruction.startswith('<b>'):
            instruction.split('\n')[0].replace('</b>', "").replace('<b>', "").strip()
            instruction = instruction.replace("*", "").strip()
            class_num = 3
        elif assistant_match:
            instruction = instruction[:assistant_match.start()].strip()
            instruction = instruction.replace("**", "").strip() if instruction.find("**") == 1 else instruction.strip()
            class_num = 4
        elif step_match:
            instruction = instruction[:step_match.start()].strip()
            instruction = instruction.replace("**", "").strip() if instruction.find("**") == 1 else instruction.strip()
            class_num = 5
        elif instruction.split('\n')[0].strip().endswith(':'):
            colon_pos = instruction.split('\n')[0].strip().rfind(':')
            if '#' in instruction:
                instruction = instruction.split('#')[0].strip()
                instruction = instruction.replace("**", "").strip() if instruction.find("**") == 1 else instruction.strip()
                class_num = 6.1
            elif '?' in instruction:
                instruction = instruction.split('?')[0].strip() + '?'
                instruction = instruction.replace("**", "").strip() if instruction.find("**") == 1 else instruction.strip()
                class_num = 6.2
            else:
                instruction = instruction.split('\n')[0].strip()
                instruction = instruction.replace("*", "").strip()
                class_num = 6.3
        else:
            if '?' in instruction:
                instruction = instruction.split('?')[0].strip() + '?'
                instruction = instruction.replace("**", "").strip() if instruction.find("**") == 1 else instruction.strip()
                class_num = 99.1
            else:
                instruction = instruction.split('\n')[0].strip()
                instruction = instruction.replace("*", "").strip()
                class_num = 99.2

        # Remove prefixes again
        instruction = remove_prefix(instruction)

        return instruction, class_num
    
    else:
        return instruction, 0


# Define logits processor for llama-3.1 for de-markdown
def de_md_logits_processor_for_llama3_1(token_ids, logits):
    # Only process the initial logits
    if len(token_ids) == 0:
        logits[2] = -9999.999 # "#": 2,
        logits[567] = -9999.999 # "##": 567,
        logits[14711] = -9999.999 # "###": 14711,
        logits[827] = -9999.999 # "####": 827,

    return logits


# Define logits processor for flaming initial tokens
def flaming_tokens(token_ids, logits):
    # Only process the initial logits
    if len(token_ids) == 0:
        # Slightly increase the temperature for the first token
        logits = logits / 1.2
    
    return logits