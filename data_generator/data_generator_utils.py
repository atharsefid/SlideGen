import json
import jsonlines
import math
import nltk
import numpy as np
import os
import rouge
import string
import sys
from collections import defaultdict
from glob import glob
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from scipy.spatial.distance import cosine

sys.path.insert(0, '..')
from data_loader import read_pdf, read_ppt_tika_xml, read_ppt_xml


class Bart_data_generator:

    def __init__(self):
        self.stopset = set(stopwords.words('english')).union(string.punctuation)
        self.wordnet_lemmatizer = WordNetLemmatizer()
        self.snowball_stemmer = SnowballStemmer('english')
        self.rouge = rouge.Rouge()
        self.covered_sents = 0
        self.needed_sents = 0
        self.evaluator = rouge.Rouge(metrics=['rouge-n'],
                                     max_n=2,
                                     limit_length=False,
                                     apply_avg=True,
                                     apply_best=None,
                                     alpha=0.5,  # Default F1_score
                                     weight_factor=1.2,
                                     stemming=True)
        self.total_score = 0

    def normalize_tokens(self, text):
        tokens = word_tokenize(text)
        tokens = [self.snowball_stemmer.stem(token.lower()) for token in tokens if token.lower() not in self.stopset]
        return tokens

    def normalize(self, text):
        tokens = word_tokenize(text)
        tokens = [self.wordnet_lemmatizer.lemmatize(token.lower(), pos='n') for token in tokens if
                  token.lower() not in self.stopset]
        return ' '.join(tokens)

    def cosine_similarity(self, a, b):
        a = a.toarray()[0]
        b = b.toarray()[0]
        return np.dot(a, b) / (self.l2_norm(a) * self.l2_norm(b))

    @staticmethod
    def jaccard_similarity(tokens1, tokens2):
        if len(tokens1.union(tokens2)) == 0:
            return 0
        return len(tokens1.intersection(tokens2)) / len(tokens1.union(tokens2))

    @staticmethod
    def ngram(t1):
        gram123 = set()
        gram123 = gram123.union(t1)
        bi_gram = set(nltk.bigrams(t1))
        gram123 = gram123.union(bi_gram)
        three_gram = set(nltk.ngrams(t1, 3))
        gram123 = gram123.union(three_gram)
        return gram123

    @staticmethod
    def read_ppt_txt(pptfile):
        ppttxt = open(pptfile, 'r', encoding='utf-8', errors='ignore')
        slides = []
        slide = ''
        for line in ppttxt:
            if line.startswith('\f'):
                slides.append(slide)
                slide = line[1:]
            else:
                slide = slide + line
        return slides

    @staticmethod
    def generate_data_section_slide_tika():
        train_source_file = jsonlines.open('train_section_slides_t5.json', mode='w')
        test_source_file = jsonlines.open('test_section_slides_t5.json', mode='w')
        val_source_file = jsonlines.open('val_section_slides_t5.json', mode='w')

        for i in range(4500):
            print(i)
            pdf_file = glob('../../ppt_generation/slide_generator_data/data/' + str(i) + '/grobid/*.tei.xml')[0]
            ppt_tika_file = '../../ppt_generation/slide_generator_data/data/' + str(i) + '/slide.clean_tika.xml'
            slides = ['\t'.join(slide).replace('\n', '\t') for slide in read_ppt_tika_xml(ppt_tika_file)]
            map_file = '../../ppt_generation/slide_generator_data/data/' + str(i) + '/slide_section_map.json'
            sections = ['\t'.join(section).replace('\n', '\t') for section in read_pdf(pdf_file)]

            try:
                with open(map_file, 'r') as f:
                    mappings = json.load(f)
            except FileNotFoundError as e:
                print(i, e.strerror)
                continue
            if 0 <= i < 4000:
                out_file = train_source_file
            elif 4000 <= i < 4250:
                out_file = val_source_file
            elif 4250 <= i < 4500:
                out_file = test_source_file
            # print(len(sections), len(slides), mappings)
            for section_id, section in enumerate(sections):
                summary = []
                for key, val in mappings.items():
                    if int(val) == section_id:
                        summary.append(slides[int(key)])
                summary = '\n'.join(summary)
                if i <= 4250:
                    if len(summary) > 10:
                        out_file.write({"text": 'summarize: ' + section, "summary": summary})
                else:
                    out_file.write({"text": 'summarize: ' + section, "summary": summary})

    @staticmethod
    def generate_data_section_slide():
        train_source_file = jsonlines.open('train_section_slides_CLEAN.json', mode='w')
        test_source_file = jsonlines.open('test_section_slides_CLEAN.json', mode='w')
        val_source_file = jsonlines.open('val_section_slides_CLEAN.json', mode='w')
        tika_count = 0
        for i in range(4500):
            print(i)
            pdf_file = glob('../../ppt_generation/slide_generator_data/data/' + str(i) + '/grobid/*.tei.xml')[0]
            ppt_file = '../../ppt_generation/slide_generator_data/data/' + str(i) + '/layered_slide.xml'
            if os.path.isfile(ppt_file):
                slides = ['\t'.join(slide).replace('\n', '\t') for slide in read_ppt_xml(ppt_file)]
            else:
                tika_count += 1
                ppt_tika_file = '../../ppt_generation/slide_generator_data/data/' + str(i) + '/slide.clean_tika.xml'
                slides = ['\t'.join(slide).replace('\n', '\t') for slide in read_ppt_tika_xml(ppt_tika_file)]
            map_file = '../../ppt_generation/slide_generator_data/data/' + str(i) + '/slide_section_map.json'
            sections = ['\t'.join(section).replace('\n', '\t') for section in read_pdf(pdf_file)]

            try:
                with open(map_file, 'r') as f:
                    mappings = json.load(f)
            except FileNotFoundError as e:
                print(i, e.strerror)
                continue
            if 0 <= i < 4000:
                out_file = train_source_file
            elif 4000 <= i < 4250:
                out_file = val_source_file
            elif 4250 <= i < 4500:
                out_file = test_source_file
            # print(len(sections), len(slides), mappings)
            empty_section_count = 0  # we will allow maximum of two empty sections for each paper
            for section_id, section in enumerate(sections):
                summary = []
                for key, val in mappings.items():
                    if int(val) == section_id:
                        summary.append(slides[int(key)])
                summary = '\n'.join(summary)
                if i <= 4250:
                    if len(summary) > 10:
                        out_file.write({"text": section, "summary": summary})
                    elif len(summary) <= 10 and empty_section_count < 2:
                        empty_section_count += 1
                        out_file.write({"text": section, "summary": summary})
                else:
                    out_file.write({"text": section, "summary": summary})

        print('Total tika xmls being used are:', tika_count)


dg = Bart_data_generator()
dg.generate_data_section_slide_tika()
