import argparse
import glob
import random

import numpy as np
from datasets import load_metric

from data_loader import *

random.seed(2019)
np.random.seed(2019)


def calculate_rouge(candidates, references):
    print('There are %d reference summaries and %d candidate summaries.' % (len(references), len(candidates)))
    metric = load_metric('rouge')
    metric.add_batch(predictions=candidates, references=references)
    rouge_scores = metric.compute()
    return rouge_scores


def rouge_results_to_str(results_dict):
    return ">> ROUGE-F(1/2/3/l): {:.2f}/{:.2f}/{:.2f}\nROUGE-R(1/2/3/l): {:.2f}/{:.2f}/{:.2f}\n".format(
        results_dict['rouge1'].high.fmeasure * 100,
        results_dict['rouge2'].high.fmeasure * 100,
        results_dict['rougeL'].high.fmeasure * 100,

        results_dict['rouge1'].high.recall * 100,
        results_dict['rouge2'].high.recall * 100,
        results_dict['rougeL'].high.recall * 100

    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-summaries_file", default='temp/tst-summarization/test_generations.txt', type=str,
                        help="The name of the file containing section summaries")
    args = parser.parse_args()

    section_summaries = open(args.summaries_file, 'r').readlines()
    start_index = 0
    candidates = []
    references = []
    giant_summary_count = 0
    for i in range(4250, 4500):
        print(i)
        pdf_file = glob.glob('raw_data/' + str(i) + '/*.tei.xml')[0]
        ppt_file = glob.glob('raw_data/' + str(i) + '/*.clean_tika.xml')[0]
        map_file = glob.glob('raw_data/' + str(i) + '/*.section_map.json')[0]
        sections = ['\t'.join(section).replace('\n', '\t') for section in read_pdf(pdf_file)]
        article = ' '.join(sections)
        slides = ['\t'.join(slide).replace('\n', '\t') for slide in read_ppt_tika_xml(ppt_file)]
        summary = ''.join(section_summaries[start_index: start_index + len(sections)])
        reference = ' '.join(slides)
        references.append(reference.strip())
        # limit length of summary to 20% of the length of document
        limit = int(0.2 * len(article))
        if len(summary) > limit:
            print(len(summary), limit)
            summary = summary[:limit]
            giant_summary_count += 1
        candidates.append(summary.strip())
        start_index += len(sections)
    assert len(candidates) == len(
        references), f"There are {len(candidates)} candidates and {len(references)} references"
    print('Number of giant summaries are: ', giant_summary_count)

    result = rouge_results_to_str(calculate_rouge(candidates, references))
    print(result)
