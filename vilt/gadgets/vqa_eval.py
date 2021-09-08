# coding=utf-8

__author__ = 'aagrawal'

# This code is based on the code written by Tsung-Yi Lin for MSCOCO Python API available at the following link: 
# (https://github.com/tylin/coco-caption/blob/master/pycocoevalcap/eval.py).
import sys
import re


class VQAEval:
    def __init__(self, vqa, vqa_res, n=2):
        self.n = n
        self.accuracy = {}
        self.eval_qa = {}
        self.eval_ques_type = {}
        self.eval_ans_type = {}
        self.vqa = vqa
        self.vqa_res = vqa_res
        self.params = {'question_id': vqa.get_ques_ids()}
        self.contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
                             "couldnt": "couldn't", \
                             "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't",
                             "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't", \
                             "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't",
                             "hed": "he'd", "hed've": "he'd've", \
                             "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's",
                             "Id've": "I'd've", "I'dve": "I'd've", \
                             "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've",
                             "it'dve": "it'd've", "itll": "it'll", "let's": "let's", \
                             "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've",
                             "mightn'tve": "mightn't've", "mightve": "might've", \
                             "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've",
                             "oclock": "o'clock", "oughtnt": "oughtn't", \
                             "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't",
                             "shed've": "she'd've", "she'dve": "she'd've", \
                             "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't",
                             "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", \
                             "somebody'd": "somebodyd", "somebodyd've": "somebody'd've",
                             "somebody'dve": "somebody'd've", "somebodyll": "somebody'll", \
                             "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've",
                             "someone'dve": "someone'd've", \
                             "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd",
                             "somethingd've": "something'd've", \
                             "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's",
                             "thered": "there'd", "thered've": "there'd've", \
                             "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd",
                             "theyd've": "they'd've", \
                             "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've",
                             "twas": "'twas", "wasnt": "wasn't", \
                             "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't",
                             "whatll": "what'll", "whatre": "what're", \
                             "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd",
                             "wheres": "where's", "whereve": "where've", \
                             "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll",
                             "whos": "who's", "whove": "who've", "whyll": "why'll", \
                             "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've",
                             "wouldnt": "wouldn't", "wouldnt've": "wouldn't've", \
                             "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll",
                             "yall'd've": "y'all'd've", \
                             "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd",
                             "youd've": "you'd've", "you'dve": "you'd've", \
                             "youll": "you'll", "youre": "you're", "youve": "you've"}
        self.manual_map = {'none': '0',
                           'zero': '0',
                           'one': '1',
                           'two': '2',
                           'three': '3',
                           'four': '4',
                           'five': '5',
                           'six': '6',
                           'seven': '7',
                           'eight': '8',
                           'nine': '9',
                           'ten': '10'
                           }
        self.articles = ['a',
                         'an',
                         'the'
                         ]

        self.period_strip = re.compile("(?!<=\d)(\.)(?!\d)")
        self.comma_strip = re.compile("(\d)(\,)(\d)")
        self.punct = [';', r"/", '[', ']', '"', '{', '}',
                      '(', ')', '=', '+', '\\', '_', '-',
                      '>', '<', '@', '`', ',', '?', '!']

    def evaluate(self, ques_ids=None):
        if ques_ids is None:
            ques_ids = [quesId for quesId in self.params['question_id']]
        gts = {}
        res = {}
        for quesId in ques_ids:
            gts[quesId] = self.vqa.qa[quesId]
            res[quesId] = self.vqa_res.qa[quesId]

        # =================================================
        # Compute accuracy
        # =================================================
        acc_qa = []
        acc_ques_type = {}
        acc_ans_type = {}
        print("computing accuracy")
        step = 0
        for quesId in ques_ids:
            res_ans = res[quesId]['answer']
            res_ans = res_ans.replace('\n', ' ')
            res_ans = res_ans.replace('\t', ' ')
            res_ans = res_ans.strip()
            res_ans = self.process_punctuation(res_ans)
            res_ans = self.process_digit_article(res_ans)
            gt_acc = []
            gt_answers = [ans['answer'] for ans in gts[quesId]['answers']]
            if len(set(gt_answers)) > 1:
                for ansDic in gts[quesId]['answers']:
                    ansDic['answer'] = self.process_punctuation(ansDic['answer'])
            for gtAnsDatum in gts[quesId]['answers']:
                other_gt_ans = [item for item in gts[quesId]['answers'] if item != gtAnsDatum]
                matching_ans = [item for item in other_gt_ans if item['answer'] == res_ans]
                acc = min(1, float(len(matching_ans)) / 3)
                gt_acc.append(acc)
            ques_type = gts[quesId]['question_type']
            ans_type = gts[quesId]['answer_type']
            avg_gt_acc = float(sum(gt_acc)) / len(gt_acc)
            acc_qa.append(avg_gt_acc)
            if ques_type not in acc_ques_type:
                acc_ques_type[ques_type] = []
            acc_ques_type[ques_type].append(avg_gt_acc)
            if ans_type not in acc_ans_type:
                acc_ans_type[ans_type] = []
            acc_ans_type[ans_type].append(avg_gt_acc)
            self.set_eval_qa(quesId, avg_gt_acc)
            self.set_eval_ques_type(quesId, ques_type, avg_gt_acc)
            self.set_eval_ans_type(quesId, ans_type, avg_gt_acc)
            if step % 100 == 0:
                self.update_progress(step / float(len(ques_ids)))
            step = step + 1

        self.set_accuracy(acc_qa, acc_ques_type, acc_ans_type)
        print("Done computing accuracy")

    def process_punctuation(self, in_text):
        out_text = in_text
        for p in self.punct:
            if (p + ' ' in in_text or ' ' + p in in_text) or (re.search(self.comma_strip, in_text) is not None):
                out_text = out_text.replace(p, '')
            else:
                out_text = out_text.replace(p, ' ')
        out_text = self.period_strip.sub("",
                                         out_text,
                                         re.UNICODE)
        return out_text

    def process_digit_article(self, in_text):
        out_text = []
        temp_text = in_text.lower().split()
        for word in temp_text:
            word = self.manual_map.setdefault(word, word)
            if word not in self.articles:
                out_text.append(word)
            else:
                pass
        for wordId, word in enumerate(out_text):
            if word in self.contractions:
                out_text[wordId] = self.contractions[word]
        out_text = ' '.join(out_text)
        return out_text

    def set_accuracy(self, acc_qa, acc_ques_type, acc_ans_type):
        self.accuracy['overall'] = round(100 * float(sum(acc_qa)) / len(acc_qa), self.n)
        self.accuracy['per_question_type'] = {
            quesType: round(100 * float(sum(acc_ques_type[quesType])) / len(acc_ques_type[quesType]), self.n) for
            quesType
            in acc_ques_type
        }
        self.accuracy['per_answer_type'] = {
            ansType: round(100 * float(sum(acc_ans_type[ansType])) / len(acc_ans_type[ansType]), self.n) for ansType in
            acc_ans_type
        }

    def set_eval_qa(self, ques_id, acc):
        self.eval_qa[ques_id] = round(100 * acc, self.n)

    def set_eval_ques_type(self, ques_id, ques_type, acc):
        if ques_type not in self.eval_ques_type:
            self.eval_ques_type[ques_type] = {}
        self.eval_ques_type[ques_type][ques_id] = round(100 * acc, self.n)

    def set_eval_ans_type(self, ques_id, ans_type, acc):
        if ans_type not in self.eval_ans_type:
            self.eval_ans_type[ans_type] = {}
        self.eval_ans_type[ans_type][ques_id] = round(100 * acc, self.n)

    def update_progress(self, progress):
        bar_length = 20
        status = ""
        if isinstance(progress, int):
            progress = float(progress)
        if not isinstance(progress, float):
            progress = 0
            status = "error: progress var must be float\r\n"
        if progress < 0:
            progress = 0
            status = "Halt...\r\n"
        if progress >= 1:
            progress = 1
            status = "Done...\r\n"
        block = int(round(bar_length * progress))
        text = "\rFinished Percent: [{0}] {1}% {2}".format("#" * block + "-" * (bar_length - block),
                                                           int(progress * 100), status)
        sys.stdout.write(text)
        sys.stdout.flush()