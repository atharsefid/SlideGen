import glob
import json
import math
import numpy as np
import torch
import traceback
from multiprocessing import Lock
from multiprocessing import Pool
from scipy import spatial
from transformers import AutoTokenizer, AutoModel

from data_loader import *


class Embedder:

    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def embed_sentences(self, sentences, device, maxlen):

        device = torch.device("cuda:" + str(device))
        # Load AutoModel from huggingface model repository
        tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/paraphrase-TinyBERT-L6-v2")
        model = AutoModel.from_pretrained("sentence-transformers/paraphrase-TinyBERT-L6-v2").to(device)

        # tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
        # model = AutoModel.from_pretrained("facebook/bart-large").to(device)

        # Tokenize sentences
        encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=maxlen, return_tensors='pt')
        encoded_input.to(device)

        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)

        # Perform pooling. In this case, mean pooling
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return sentence_embeddings

    def embed_paragraph(self, paragraph):
        sentence_embeddings = self.embed_sentences(paragraph)
        # print('sentence embedding shape:', sentence_embeddings.shape)
        paragraph_embedding = torch.mean(sentence_embeddings, 0)
        # print('document embedding shape:', paragraph_embedding.shape)
        return paragraph_embedding

    def embed_document(self, paragraphs, device, document='paper'):
        paragraph_sent_count = []
        sentences = []
        if document == 'paper':
            maxlen = 512
        elif document == 'slide':
            maxlen = 256

        for paragraph in paragraphs:
            paragraph_sent_count.append(len(paragraph))
            sentences.extend(paragraph)
        chunk_size = 100
        sentence_embeddings = torch.cat([self.embed_sentences(sentences[i:i + chunk_size], device, maxlen) \
                                         for i in range(0, len(sentences), chunk_size)], dim=0)

        paragraph_embeddings = []
        start = 0
        for paragraph_length in paragraph_sent_count:
            paragraph_embeddings.append(torch.mean(sentence_embeddings[start:start + paragraph_length], dim=0))
            start += paragraph_length
        return paragraph_embeddings


def max_similarity_map(similarities):
    mapping = {}
    for i in range(len(similarities)):
        mapping[i] = int(np.argmax(similarities[i]))
    # print(mapping)
    return mapping


def edit_distance_map(similarities):
    scores = [[0] * len(similarities[0]) for _ in range(len(similarities))]
    # dp base
    for i in range(len(similarities)):
        scores[i][0] = similarities[i][0]
    for j in range(len(similarities[0])):
        scores[0][j] = similarities[0][j]
    mapping = {}
    for i in range(1, len(similarities)):
        for j in range(1, len(similarities[0])):
            scores[i][j] = max(scores[i - 1][j],  # slide i is not matched
                               scores[i][j - 1],  # section j is not matches
                               scores[i - 1][j - 1] + round(similarities[i][j], 4))

    i, j = len(similarities) - 1, len(similarities[0]) - 1
    while True:
        # print(i,j)
        if i == 0 or j == 0:
            break
        if scores[i][j] == scores[i - 1][j]:
            i -= 1
        elif scores[i][j] == scores[i][j - 1]:
            j -= 1
        elif scores[i][j] == scores[i - 1][j - 1] + round(similarities[i][j], 4):
            mapping[i] = j
            i -= 1
            j -= 1
    # print('--------- similarities')
    # for i in range(len(similarities)):
    #     print(similarities[i])
    # print('--------- scores')
    # for i in range(len(scores)):
    #     print(scores[i])
    # print('---------- mapping')
    # print(mapping)
    return mapping


def match(slide_embeddings, section_embeddings):
    similarity_matrix = [[0] * len(section_embeddings) for _ in range(len(slide_embeddings))]
    for i_index, slide_embedding in enumerate(slide_embeddings):
        slide_embedding = slide_embedding.cpu().detach().numpy()
        for j_index, section_embedding in enumerate(section_embeddings):
            section_embedding = section_embedding.cpu().detach().numpy()
            similarity = 1 - spatial.distance.cosine(slide_embedding, section_embedding)
            similarity_matrix[i_index][j_index] = round(similarity, 4)
    # slide_section_map = edit_distance_map(similarity_matrix)
    slide_section_map = max_similarity_map(similarity_matrix)

    return slide_section_map


def slide_section_match(argument):
    try:
        pdf_file, ppt_file, index, device = argument[0], argument[1], argument[2], argument[3]
        print('device is: ', device, pdf_file)
        embedder = Embedder()
        sections = [section for section in read_pdf(pdf_file)]
        slides = read_ppt_tika_xml(ppt_file)
        section_embeds = embedder.embed_document(sections, device)
        slide_embeds = embedder.embed_document(slides, device)
        mapping = match(slide_embeds, section_embeds)
        assert len(mapping) == len(slides)
        with open('raw_data/' + str(index) + '/'+str(index) + '.section_map.json', 'w') as map_file:
            json.dump(mapping, map_file)
    except Exception as e:
        print('-- Exception', index, e)


def process_range(input):
    range_lower, range_upper, device = input
    for task_id in range(range_lower, range_upper):
        pdf_file = glob.glob('raw_data/' + str(task_id) + '/*.tei.xml')[0]
        ppt_file = glob.glob('raw_data/' + str(task_id) + '/*.clean_tika.xml')[0]
        slide_section_match((pdf_file, ppt_file, task_id, device))


import logging
from multiprocessing import Pool

def proc_wrapper(func, *args, **kwargs):
    """Print exception because multiprocessing lib doesn't return them right."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logging.exception(e)
        raise
def callback(result):
    print('success', result)

def callback_error(result):
    print('-----error', result)


ranges = [(0, 1000, 0), (1000, 2000, 1), (2000, 3000, 2), (3000, 4500, 3)]

pool = Pool(4)  # since there are 4 gpus
result = pool.imap(process_range, ranges)
for r in result:
    if isinstance(r, Exception):
        print("Got exception: {}".format(result))
pool.close()
pool.join()
print('Finished processing all files.')

# process_range(1,1000, 0)