# Fliplet support Chatbot

This a test script that : 

1 - Crawls through one or more domains and extracts text from them (Stored in flat .txt files)
2 - Creates openAI tokens using the openAI library Tiktoken
3 - Organizes these tokens as embeddings in a CSV file "embeddings.csv"
4 - Creates the context for a question usings these embeddings: Analyses the embeddings and determines the most relevant tokens to attach to the question as context
5 - Sends a prompt with a context to the openAI model "gpt-3.5-turbo-0301" 

Once the web crawling is done and the embeddings file is read, you can use test.py to only send and receive prompts.

## Prerequisites

- Install python on your machine
- Some familiarity with a code editor

## Instructions

There are two scripts that we're going to run: 

- web-qa.py : This script crawls user specified domains, creates the embeddings and asks sample questions
- test.py: This script can be modified with questions and other prompt settings to test responses. 

Please follow  these steps to get started: 

1. Copy/clone this folder into your machine
2. Check if python is installed in your machine by typing in your terminal  `python --version`
3. Open the file "web-qa.py" and "test.py" in your code editor (VScode)
4. Copy your openAI key in line 17 for both files `openai.api_key = "openAI API key"` in both files. (You need to attach a credit card to your account, but it's very cheap when used)
5. in "web-qa.py", add  the domains that you want to crawl at line 22: `domains = ["domain.com","domain2.com"]`
6. You can add sample questions at line 384
7. in your terminal, open a jupyter notebook by typing in your terminal: `jupyter notebook`
8. It'll open a jupyter UI in your browser. Go to new -> terminal
9. Create a python environment:  `python -m venv env`
10. Activate this environment :  `source env/bin/activate`
11. Install the required packages for the environment: `pip install -r requirements.txt`
12. Run the web crawl script: `python web-qa.py` (It'll take some time)
13. open "test.py" in your code editor and write your sample questions starting line 104 : 
    `print(answer_question(df, question="question"))`
14. run the test script in your jupyter terminal: python `test.py`

## Experimentations 

### temperature

The requests for the openAI are already a bit tuned, but you can still work on the "temperature" parameter. 

The parameter is between 0 and 1 where 0 is no risk at all while building the response and a temperature of 1 will make the AI hallucinate responses. 

It's currently set at 0.5, you can change it at line 86 of the file "test.py"

### Stance and prompt

The bot is currently configured with an "assistant" stance, it'll try to answer questions the best it can within this stance and temperature. 

You can change that in line 89. 

This code defines how we interact with the bot: 

`messages=[{"role": "assistant", "content": f"Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:"}]`

The "role" part of this code defines the stance of the bot. 

You can also modify the static part of this prompt in order to define how the bot behaves when it doesn't know the answer to a prompt. 
