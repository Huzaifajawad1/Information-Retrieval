import collections
import math
import os
from collections import defaultdict
from typing import List, Dict, Set
import nltk
from nltk.stem.porter import PorterStemmer
from bs4 import BeautifulSoup
import string
import re
import tkinter as tk
import pickle


# nltk.download('punkt')  # Only required once


def check_format(s):
    pattern = r'^\w+\s+\w+\s+/\d+$'  # regular expression pattern
    return bool(re.match(pattern, s))


def preprocess_document(text: str, stopwords: List[str]) -> List[str]:
    soup = BeautifulSoup(text, "html.parser")
    text = soup.getText()
    text = text.replace("-", " ")
    # Remove all punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenize the text into individual words
    words = nltk.word_tokenize(text.lower())

    # Remove stop words
    stop_words = set(stopwords)
    words = [word for word in words if word not in stop_words]

    # Apply stemming using the Porter stemmer
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]

    return words


def return_posting(word) -> List[int]:
    return [doc_id for doc_id in list(positional_index[word].keys()) if doc_id is not None]


def query_break(infix_tokens):
    """ Parse Query    """
    precedence = {'NOT': 3, 'AND': 2, 'OR': 1, '(': 0, ')': 0}

    output = []
    operator_stack = []

    for token in infix_tokens:
        if token == '(':
            operator_stack.append(token)

        # if right bracket, pop all operators from operator stack onto output until we hit left bracket
        elif token == ')':
            operator = operator_stack.pop()
            while operator != '(':
                output.append(operator)
                operator = operator_stack.pop()

        # if operator, pop operators from operator stack to queue if they are of higher precedence
        elif token in precedence:
            # if operator stack is not empty
            if operator_stack:
                current_operator = operator_stack[-1]
                while operator_stack and precedence[current_operator] > precedence[token]:
                    output.append(operator_stack.pop())
                    if operator_stack:
                        current_operator = operator_stack[-1]
            operator_stack.append(token)  # add token to stack
        else:
            output.append(token.lower())

    # while there are still operators on the stack, pop them into the queue
    while operator_stack:
        output.append(operator_stack.pop())

    return output


def and_op(left_operand, right_operand):
    # perform 'merge'
    result = []  # results list to be returned
    l_index = 0  # current index in left_operand
    r_index = 0  # current index in right_operand
    l_skip = int(math.sqrt(len(left_operand)))  # skip pointer distance for l_index
    r_skip = int(math.sqrt(len(right_operand)))  # skip pointer distance for r_index

    while l_index < len(left_operand) and r_index < len(right_operand):
        l_item = left_operand[l_index]  # current item in left_operand
        r_item = right_operand[r_index]  # current item in right_operand

        # case 1: if match
        if l_item == r_item:
            result.append(l_item)  # add to results
            l_index += 1  # advance left index
            r_index += 1  # advance right index

        # case 2: if left item is more than right item
        elif l_item > r_item:
            # if r_index can be skipped (if new r_index is still within range and resulting item is <= left item)
            if (r_index + r_skip < len(right_operand)) and right_operand[r_index + r_skip] <= l_item:
                r_index += r_skip
            # else advance r_index by 1
            else:
                r_index += 1

        # case 3: if left item is less than right item
        else:
            # if l_index can be skipped (if new l_index is still within range and resulting item is <= right item)
            if (l_index + l_skip < len(left_operand)) and left_operand[l_index + l_skip] <= r_item:
                l_index += l_skip
            # else advance l_index by 1
            else:
                l_index += 1

    return result


def or_op(left_operand, right_operand):
    result = []  # union of left and right operand
    l_index = 0  # current index in left_operand
    r_index = 0  # current index in right_operand

    # while lists have not yet been covered
    while l_index < len(left_operand) or r_index < len(right_operand):
        # if both list are not yet exhausted
        if l_index < len(left_operand) and r_index < len(right_operand):
            l_item = left_operand[l_index]  # current item in left_operand
            r_item = right_operand[r_index]  # current item in right_operand

            # case 1: if items are equal, add either one to result and advance both pointers
            if l_item == r_item:
                result.append(l_item)
                l_index += 1
                r_index += 1

            # case 2: l_item greater than r_item, add r_item and advance r_index
            elif l_item > r_item:
                result.append(r_item)
                r_index += 1

            # case 3: l_item lower than r_item, add l_item and advance l_index
            else:
                result.append(l_item)
                l_index += 1

        # if left_operand list is exhausted, append r_item and advance r_index
        elif l_index >= len(left_operand):
            r_item = right_operand[r_index]
            result.append(r_item)
            r_index += 1

        # else if right_operand list is exhausted, append l_item and advance l_index
        else:
            l_item = left_operand[l_index]
            result.append(l_item)
            l_index += 1

    return result


def not_op(right_operand, entire_docs):
    # complement of an empty list is list of all indexed docIDs
    if not right_operand:
        return entire_docs

    result = []
    r_index = 0  # index for right operand
    for item in entire_docs:
        # if item do not match that in right_operand, it belongs to compliment
        if item != right_operand[r_index]:
            result.append(item)
        # else if item matches and r_index still can progress, advance it by 1
        elif r_index + 1 < len(right_operand):
            r_index += 1
    return result


def process_query(query):
    # prepare query list
    query = query.replace('(', '( ')
    query = query.replace(')', ' )')
    query = query.split(' ')

    entire_docs = list(range(1, 31))

    results_stack = []
    postfix_queue = collections.deque(query_break(query))  # get query in postfix notation as a queue

    while postfix_queue:
        token = postfix_queue.popleft()
        result = []  # the evaluated result at each stage
        # if operand, add postings list for term to results stack
        if token != 'AND' and token != 'OR' and token != 'NOT':
            stemmer = PorterStemmer()
            token = stemmer.stem(token)
            # default empty list if not in dictionary
            if token in positional_index:
                result = return_posting(token)

        elif token == 'AND':
            right_operand = results_stack.pop()
            left_operand = results_stack.pop()
            result = and_op(left_operand, right_operand)  # evaluate AND

        elif token == 'OR':
            right_operand = results_stack.pop()
            left_operand = results_stack.pop()
            result = or_op(left_operand, right_operand)  # evaluate OR

        elif token == 'NOT':
            right_operand = results_stack.pop()
            result = not_op(right_operand, entire_docs)  # evaluate NOT

        results_stack.append(result)

        # NOTE: at this point results_stack should only have one item ,and it is the final result
    if len(results_stack) != 1:
        print("ERROR: Invalid Query. Please check query syntax.")  # check for errors
        return None

    return results_stack.pop()


def proximity_query(query: str) -> Set[int]:
    """
    Performs a proximity query on a positional index.
    """
    words, k = query.split('/')
    word1, word2 = words.split()
    k = int(k)
    stemmer = PorterStemmer()
    word1 = stemmer.stem(word1)
    word2 = stemmer.stem(word2)
    if word1 not in positional_index or word2 not in positional_index:
        return set()

    result = set()

    # Iterate over all documents containing both words
    for doc_id in set(positional_index[word1].keys()) & set(positional_index[word2].keys()):
        positions1 = positional_index[word1][doc_id]
        positions2 = positional_index[word2][doc_id]

        # Use a nested loop to check all pairs of positions within k words of each other
        for i in positions1:
            for j in positions2:
                if abs(i - j) <= k:
                    result.add(doc_id)
                    break  # If we find one pair of positions that satisfy the proximity constraint, we can stop
                    # searching for this document

    return result


def create_positional_index(documents: List[str], positional_index,doc_number: int):
    word_position = 0
    for word in documents:
        word_position += 1
        if not positional_index[word][doc_number] or positional_index[word][doc_number][-1] != word_position:
            positional_index[word][doc_number].append(word_position)


def default_dict():
    return defaultdict(list)


def code():
    # Creating an array which has all stopwords
    """stopwords = []
    with open("Stopword-List.txt") as f:
        for line in f:
            word = line.strip()
            if word:
                stopwords.append(word)
    # Uses Hashing
    positional_index: Dict[str, Dict[int, List[int]]] = defaultdict(default_dict)

    # To create the index for the first time and store it into positional_index.pickle
    for i in range(1, 31):
        filename = f"Dataset/{i}.txt"
        if not os.path.isfile(filename):
            print(f"Error: could not open file {filename}")
            continue
        with open(filename) as f:
            file_contents = f.read()
        tokens = preprocess_document(file_contents, stopwords)
        create_positional_index(tokens, positional_index, i)
    file = open(r'index.pkl', 'wb')
    pickle.dump(positional_index, file)
    print("Data entered into file0")
    file.close()"""

    # reload object from file
    file2 = open(r'index.pkl', 'rb')
    positional_index = pickle.load(file2)
    print("Data read from file")
    file2.close()
    # for printing the positional index
    '''for word, doc_positions in positional_index.items():
        print(word, end=' : ')
        for doc, positions in doc_positions.items():
            print(f"Doc{doc} -> {positions}", end=' ')
        print()'''
    # query_input = input("Enter your query: ")
    '''For NOT queries the world should come before the NOT'''
    return positional_index


def answer(query_input):
    if check_format(query_input):
        Output = proximity_query(query_input)
    else:
        Output = process_query(query_input)
    if Output is not None:
        return Output
    else:
        return "Not a valid Query"


def create_widgets(root):
    # Create label for input field
    input_label = tk.Label(root, height=4, text="Enter Query=> Format should be : w1 AND/OR w2 , NOT w1 , w1 w2 /k",
                           font=("Arial", 10))
    input_label.pack()

    # Create input field
    input_field = tk.Entry(root, width=50, font=("Arial", 14))
    input_field.pack()

    # Create output field
    output_field = tk.Text(root, height=10, width=50, font=("Arial", 14))
    output_field.pack(pady=(20, 0))

    # Create button
    button = tk.Button(root, text="Submit", font=("Arial", 14), bg="blue", fg="white",
                       command=lambda: button_clicked(input_field, output_field))
    button.pack()

    # Set window size
    root.geometry("800x600")


def button_clicked(input_field, output_field):
    # Delete Previous Data
    output_field.delete(1.0, tk.END)
    # Get input from input field
    input_text = input_field.get()
    output_text = answer(input_text)
    if output_text != "Not a valid Query":
        if len(output_text) == 0:
            output_field.insert(tk.END, "No results found")
        else:
            # Update output field with processed text
            output_field.insert(tk.END, "Doc IDs :\n")
            for i in output_text:
                output_field.insert(tk.END, i)
                output_field.insert(tk.END, "   ")
    else:
        output_field.insert(tk.END, output_text)


def start_gui():
    root = tk.Tk()
    root.title("Boolean GUI")
    create_widgets(root)
    root.mainloop()


if __name__ == "__main__":
    positional_index = code()
    start_gui()
