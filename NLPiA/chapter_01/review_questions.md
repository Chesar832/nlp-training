
# Chapter 1 review questions

![banner.jpg](https://symetricx.com/wp-content/uploads/2021/03/NLP-banner.jpg)

These are questions to test my understanding from the chapter:

1. **Why is NLP considered to be a core enabling feature for AGI (humanlike
AI)?**
- Because natural language abstracts all kind of information about the world and NLP can translate that language to a "computer language" (vectors, depending of the task to perform).

2. **Why do advanced NLP models tend to show significant discriminatory
biases?**
- Because we lived in a biased world, so ours corpora are biased too. Even in some page the author says that this *bias* is resposable for certain biased vectorization where a women is more "similar" to the word "nurse" than a man.

3. **How is it possible to create a prosocial chatbot using training data from
sources that include antisocial examples?**
<<<<<<< HEAD
- Managing the bias and possibly applying external rules to the chatbot for ethical purposes.
=======
- We have to domain the topic that we're going to talk and comprehend teh challenges in the expressions of our corpus and how we will remove or avoid the typical bias in its this context.
>>>>>>> 1f082f57d62fb971d26c66bc60f496a6eabc8252

4. **What are 4 different approaches or architectures for building a chatbot?**
- Keyword-based chatbot → Returns a prepared response which is based on the words found in the message.
- Pattern-based chatbot → Works similarly to the *keyword-based* approach, but these patterns are more robust for a variety of messages.
- Search-based chatbot → Returns the reponse for the more similar question in a database of Q&A.
- Natural language chatbot → This approach uses a NLP pipeline (involves a more complex and extensive process) and returns a more human-like response.
<<<<<<< HEAD
=======

5. **How is NLP used within a search engine?**
- In a search engine, NLP is used for two purposes:
    1. Text Generation: The NLP pipeline generates text to suggest you related (similar) searches like yours.
    2. Text Summarization: The NLP pipeline compute the similarity between the prompt and all the web pages to return first the most accurate content related to your search.

6. **Write a regular expression to recognize your name and all the variations
on its spelling (including nicknames) that you’ve seen.**
- The expression is:
    ``` python
    > my_name_regular_expression = r'\b[cC]+[eE]+[sS]+[aA]+[rR]+\b'
    ```

7. **Write a regular expression to try to recognize a sentence boundary
(usually a period ("."), question mark "?", or exclamation mark "!")**
- The expression is:
    ``` python
    > pattern = r'[.!?](?=\s|$)'
    ```
>>>>>>> 1f082f57d62fb971d26c66bc60f496a6eabc8252
