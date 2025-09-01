from jupyter_ai.personas.base_persona import BasePersona, PersonaDefaults
from jupyterlab_chat.models import Message
from pocketflow import Node, Flow
from litellm import completion
import yaml
from ddgs import DDGS
import requests
import sys

## UTILS (outside the Persona class)
def search_web_duckduckgo(query):
    results = DDGS().text(query, max_results=5)
    # Convert results to a string
    results_str = "\n\n".join([f"Title: {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}" for r in results])
    return results_str

def search_web_brave(query):

    url = f"https://api.search.brave.com/res/v1/web/search?q={query}"
    api_key = "your brave search api key"

    headers = {
        "accept": "application/json",
        "Accept-Encoding": "gzip",
        "x-subscription-token": api_key
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        results = data['web']['results']
        results_str = "\n\n".join([f"Title: {r['title']}\nURL: {r['url']}\nDescription: {r['description']}" for r in results])     
    else:
        print(f"Request failed with status code: {response.status_code}")
    return results_str

## NODES : Outside the Persona class
class DecideAction(Node):
    def prep(self, shared):
        """Prepare the context and question for the decision-making process."""
        # Get the current context (default to "No previous search" if none exists)
        context = shared.get("context", "No previous search")
        # Get the question from the shared store
        question = shared["question"]
        # Return both for the exec step
        return question, context
        
    def exec(self, inputs):
        """Call the LLM to decide whether to search or answer."""
        question, context = inputs
        
        print(f"🤔 Agent deciding what to do next...")
        
        # Create a prompt to help the LLM decide what to do next with proper yaml formatting
        prompt = f"""
### CONTEXT
You are a research assistant that can search the web.
Question: {question}
Previous Research: {context}

### ACTION SPACE
[1] search
  Description: Look up more information on the web
  Parameters:
    - query (str): What to search for

[2] answer
  Description: Answer the question with current knowledge
  Parameters:
    - answer (str): Final answer to the question

## NEXT ACTION
Decide the next action based on the context and available actions.
Return your response in this format:

```yaml
thinking: |
    <your step-by-step reasoning process>
action: search OR answer
reason: <why you chose this action>
answer: <if action is answer>
search_query: <specific search query if action is search>
```
IMPORTANT: Make sure to:
1. Use proper indentation (4 spaces) for all multi-line fields
2. Use the | character for multi-line text fields
3. Keep single-line fields without the | character
"""
        
        # Call the LLM to make a decision
        response = self.persona.call_llm(prompt)
        
        # Parse the response to get the decision
        yaml_str = response.split("```yaml")[1].split("```")[0].strip()
        decision = yaml.safe_load(yaml_str)
        
        return decision
    
    def post(self, shared, prep_res, exec_res):
        """Save the decision and determine the next step in the flow."""
        # If LLM decided to search, save the search query
        if exec_res["action"] == "search":
            shared["search_query"] = exec_res["search_query"]
            print(f"🔍 Agent decided to search for: {exec_res['search_query']}")
        else:
            shared["context"] = exec_res["answer"] #save the context if LLM gives the answer without searching.
            print(f"💡 Agent decided to answer the question")
        
        # Return the action to determine the next node in the flow
        return exec_res["action"]

class SearchWeb(Node):
    def prep(self, shared):
        """Get the search query from the shared store."""
        return shared["search_query"]
        
    def exec(self, search_query):
        """Search the web for the given query."""
        # Call the search utility function
        print(f"🌐 Searching the web for: {search_query}")
        results = search_web_duckduckgo(search_query)
        return results
    
    def post(self, shared, prep_res, exec_res):
        """Save the search results and go back to the decision node."""
        # Add the search results to the context in the shared store
        previous = shared.get("context", "")
        shared["context"] = previous + "\n\nSEARCH: " + shared["search_query"] + "\nRESULTS: " + exec_res
        
        print(f"📚 Found information, analyzing results...")
        
        # Always go back to the decision node after searching
        return "decide"

class AnswerQuestion(Node):
    def prep(self, shared):
        """Get the question and context for answering."""
        return shared["question"], shared.get("context", "")
        
    def exec(self, inputs):
        """Call the LLM to generate a final answer."""
        question, context = inputs
        
        print(f"✍️ Crafting final answer...")
        
        # Create a prompt for the LLM to answer the question
        prompt = f"""
### CONTEXT
Based on the following information, answer the question.
Question: {question}
Research: {context}

## YOUR ANSWER:
Provide a comprehensive answer using the research results.
"""
        # Call the LLM to generate an answer
        answer = self.persona.call_llm(prompt)
        return answer
    
    def post(self, shared, prep_res, exec_res):
        """Save the final answer and complete the flow."""
        # Save the answer in the shared store
        shared["answer"] = exec_res
        
        print(f"✅ Answer generated successfully")
        
        # We're done - no need to continue the flow
        return "done" 


class PocketAgentPersona(BasePersona, DecideAction, SearchWeb, AnswerQuestion):
    """
    The PocketFlow persona, demo to show usage of PocketFlow API.
    """

    def __init__(self, *args, **kwargs):
        BasePersona.__init__(self, *args, **kwargs)
        DecideAction.__init__(self)
        SearchWeb.__init__(self)
        AnswerQuestion.__init__(self)

    @property
    def defaults(self):
        return PersonaDefaults(
            name="PocketAgentPersona",
            avatar_path="/api/ai/static/jupyternaut.svg",
            description="The pocketflow agent.",
            system_prompt="...",
        )
    
    ## UTILS (within the Persona class)
    def call_llm(self, prompt):
        """Calls the litellm completion API with the given messages."""
        # If messages is a string, convert it to proper message format
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]
        response = completion(
            model=self.config_manager.chat_model,
            messages=prompt,
            stream=False,
        )
        return response.choices[0].message.content
    
    async def process_message(self, message: Message):
        if not self.config_manager.chat_model:
            self.send_message(
                "No chat model is configured.\n\n"
                "You must set one first in the Jupyter AI settings, found in 'Settings > AI Settings' from the menu bar."
            )
            return
        
        ## FLOW: Initialize the flow if not already done
        decide = DecideAction()
        search = SearchWeb()
        answer = AnswerQuestion()
        decide.persona = self
        search.persona = self
        answer.persona = self
        # Connect the nodes
        # If DecideAction returns "search", go to SearchWeb
        decide - "search" >> search
        # If DecideAction returns "answer", go to AnswerQuestion
        decide - "answer" >> answer
        # After SearchWeb completes and returns "decide", go back to DecideAction
        search - "decide" >> decide
        ## Flow: Run the flow with the input message
        agent_flow = Flow(start=decide)

        ## MAIN: Answer the question from the message
        question = message.body.split(" ", 1)[1]  # Using text attribute instead of content
        shared = {"question": question}  # Using text attribute instead of content
        print(f"🤔 Processing question: {message.body.split(" ", 1)[1]}")
        agent_flow.run(shared)
        print("\n🎯 Final Answer:")
        print(shared.get("answer", "No answer found"))
        self.send_message(shared.get("answer", "No answer found"))
