import argparse
import json

from vilt.gadgets.vqa import VQA
from vilt.gadgets.vqa_eval import VQAEval


def main(args):
    with open(args.generation, 'r') as json_file:
        generated = json.load(json_file)

    vqa = VQA(args.annot_file, args.ques_file)
    vqa_res = vqa.load_res(generated, args.ques_file)
    vqa_eval = VQAEval(vqa, vqa_res, n=2)
    vqa_eval.evaluate(ques_ids=[x['question_id'] for x in generated])

    print('Validation scores')
    print('overall accuracy: {}'.format(vqa_eval.accuracy['overall']))
    for ans_type in vqa_eval.accuracy['per_answer_type']:
        print('{} accuracy: {}'.format(ans_type, vqa_eval.accuracy['per_answer_type'][ans_type]))


def parse_args():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--generation', type=str, required=True,
                        help='path to the generation file')
    parser.add_argument('--annot_file', default=None, type=str,
                        help='annotation file path for VQA (v2_mscoco_val2014_annotations.json)')
    parser.add_argument('--ques_file', default=None, type=str,
                        help='question file path for VQA (v2_OpenEnded_mscoco_val2014_questions.json)')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)