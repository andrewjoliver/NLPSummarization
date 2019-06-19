from numpy import empty
import numpy as np
import csv
import math


def clean_text(text):
    stop_words = ["a", "about", "above", "after", "again", "against", "ain", "all", "am", "an", "and", "any", "are", "aren", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "couldn", "couldn't", "d", "did", "didn", "didn't", "do", "does", "doesn", "doesn't", "doing", "don", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn", "hadn't", "has", "hasn", "hasn't", "have", "haven", "haven't", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "ll", "m", "ma", "me", "mightn", "mightn't", "more", "most", "mustn", "mustn't", "my", "myself", "needn", "needn't", "no", "nor", "not", "now", "o", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "re", "s", "same", "shan", "shan't", "she", "she's", "should", "should've", "shouldn", "shouldn't", "so", "some", "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "ve", "very", "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "won", "won't", "wouldn", "wouldn't", "y", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves", "could", "he'd", "he'll", "he's", "here's", "how's", "i'd", "i'll", "i'm", "i've", "let's", "ought", "she'd", "she'll", "that's", "there's", "they'd", "they'll", "they're", "they've", "we'd", "we'll", "we're", "we've", "what's", "when's", "where's", "who's", "why's", "would"]
    punctuation = [".", "-", "_", ",", "<", ">", "?", "/", "'", "\"", ";", ":", "[", "{", "}", "]", "\\", "|", "`", "~", "!", "@", "#", "$", "^", "&", "*", "(", ")"]
    cleaned_text = list()

    words = text.split(" ")
    for word in words:
        word = word.lower()
        for char in punctuation:
            word = word.replace(char, ' ')
                
        if word not in stop_words and len(word) > 0:
            cleaned_text.append(word)

    return cleaned_text


def clean_full(read_loc):
    gatsby = list()
    gatsby_map = list()

    with open(read_loc) as read_loc:
        lines = read_loc.readlines()
        
    for line in lines:
        if line == '\n':
            gatsby.append(" paragraph-break-here ")
        else:
            gatsby.append(line.replace('\n', ' '))

    gatsby = "".join(gatsby)
    gatsby = gatsby.split(" paragraph-break-here ")

    for line in gatsby:
        cleaned_text_res = clean_text(line)
        if len(cleaned_text_res) > 0:
            gatsby_map.append((line, cleaned_text_res))

    return gatsby_map


def cosine_similarity(paragraph1, paragraph2):
    all_words = list()
    for word in paragraph1:
        all_words.append(word)
    for word in paragraph2:
        if word not in all_words:
            all_words.append(word)
    
    all_words.sort()
    
    paragraph1vector = list()
    paragraph2vector = list()

    for word in all_words:
        paragraph1vector.append(1) if word in paragraph1 else paragraph1vector.append(0)
        paragraph2vector.append(1) if word in paragraph2 else paragraph2vector.append(0)
    
    sum = 0
    for x in range(len(paragraph1vector)):
        sum += paragraph1vector[x] * paragraph2vector[x]
    
    magnitudeA = 0
    innerSquareSum = 0
    for val in paragraph1vector:
        innerSquareSum += val*val
    magnitudeA = math.sqrt(innerSquareSum) 
    
    magnitudeB = 0
    innerSquareSum = 0
    for val in paragraph2vector:
        innerSquareSum += val*val
    magnitudeB = math.sqrt(innerSquareSum) 
    
    if (magnitudeA) == 0:
        magnitudeA = 1

    if (magnitudeB) == 0:
        magnitudeB = 1

    cosine_sim_val = float(sum) / float(magnitudeA*magnitudeB)
    
    if not isinstance(cosine_sim_val, float) or cosine_sim_val < 1e-5:
        cosine_sim_val = 0.0
    
    return cosine_sim_val


def build_matrix(paragraph_list):
    n = len(paragraph_list)
    adjacency_matrix = empty([n,n])
    
    for x in range(n):
        for y in range(n):
            cos_sim = cosine_similarity(paragraph_list[x][1], paragraph_list[y][1])
            adjacency_matrix[x][y] = cos_sim

    return adjacency_matrix


def calculate_stationary_probabilities(adjacency_matrix):
    # Code pulled from Duke University, Stats 663 from Dr. Cliburn Chan
    # http://people.duke.edu/~ccc14/sta-663-2016/homework/Homework02_Solutions.html#Part-3:-Option-2:-Using-numpy.linalg-with-transpose-to-get-the-left-eigenvectors
    P = adjacency_matrix/np.sum(adjacency_matrix, 1)[:, np.newaxis]
    P5000 = np.linalg.matrix_power(P, 5000)
    P5001 = np.dot(P5000, P)
    # check that P50 is stationary
    np.testing.assert_allclose(P5000, P5001)
    return P5001


def output_summarization_paragraphs(distribution, map, num_paragraphs):
    indices = distribution.argsort()[-(num_paragraphs):][::-1]
    for index in indices:
        print("Sentence:    " + str(map[index][0]))
        print("Probability: " + str(distribution[index]))
        print("---------------")


def main():
	gatsby_map = clean_full("../files/gatsby.txt")
	adjacency_matrix = build_matrix(gatsby_map)
	probability_distribution = calculate_stationary_probabilities(adjacency_matrix)
	output_summarization_paragraphs(probability_distribution[0], gatsby_map, 6)


if __name__ == '__main__':
	main()

