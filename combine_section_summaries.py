import glob
import numpy as np
import os
import random
import time
from pyrouge import Rouge155

random.seed(2019)
np.random.seed(2019)


def calc_rouge(candidates, references, temp_dir):
    assert len(candidates) == len(references)

    cnt = len(candidates)

    if not os.path.isdir(temp_dir):
        os.mkdir(temp_dir)
        os.mkdir(temp_dir + "/candidate")
        os.mkdir(temp_dir + "/reference")

    for i in range(cnt):
        if len(references[i]) < 1:
            continue
        with open(temp_dir + "/candidate/cand.{}.txt".format(i), "w",
                  encoding="utf-8") as f:
            f.write(candidates[i])
        with open(temp_dir + "/reference/ref.{}.txt".format(i), "w",
                  encoding="utf-8") as f:
            f.write(references[i])
    r = Rouge155()  # temp_dir=temp_dir)
    r.model_dir = temp_dir + "/reference/"
    r.system_dir = temp_dir + "/candidate/"
    r.model_filename_pattern = 'ref.#ID#.txt'
    r.system_filename_pattern = r'cand.(\d+).txt'
    rouge_results = r.convert_and_evaluate()
    results_dict = r.output_to_dict(rouge_results)
    return results_dict


from data_loader import *

# section_summaries_file = 'temp/tst-summarization/test_generations.txt' # BART
# section_summaries_file = 'temp/tst-summarization3/test_generations.txt' # BART with empty summaries
section_summaries_file = 'T5/t5_generated_summaries1.txt'  # T5-small
section_summaries = open(section_summaries_file, 'r').readlines()
start_index = 0
candidates = []
references = []
giant_summary_count = 0
for i in range(4250, 4500):
    print(i)
    pdf_file = glob.glob('../ppt_generation/slide_generator_data/data/' + str(i) + '/grobid/*.tei.xml')[0]
    ppt_file = '../ppt_generation/slide_generator_data/data/' + str(i) + '/slide.clean_tika.xml'
    map_file = '../ppt_generation/slide_generator_data/data/' + str(i) + '/slide_section_map.json'
    sections = ['\t'.join(section).replace('\n', '\t') for section in read_pdf(pdf_file)]
    article = ' '.join(sections)
    slides = ['\t'.join(slide).replace('\n', '\t') for slide in read_ppt_tika_xml(ppt_file)]
    summary = ''.join(section_summaries[start_index: start_index + len(sections)])
    reference = ' '.join(slides)
    references.append(reference)
    # limit length of summary to 20% of the length of document
    limit = int(0.2 * len(article))
    if len(summary) > limit:
        print(len(summary), limit)
        summary = summary[:limit]
        giant_summary_count += 1
    candidates.append(summary)
    start_index += len(sections)

print('Number of giant summaries are: ', giant_summary_count)
print('The last index of test data should start with 87..', start_index)
print('can len, ref len', len(candidates), len(references))
result = calc_rouge(candidates, references, 'rouge_temp')
print(result)
