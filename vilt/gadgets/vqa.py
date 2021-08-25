__author__ = 'aagrawal'
__version__ = '0.9'

# Interface for accessing the VQA dataset.

# This code is based on the code written by Tsung-Yi Lin for MSCOCO Python API available at the following link:
# (https://github.com/pdollar/coco/blob/master/PythonAPI/pycocotools/coco.py).

# The following functions are defined:
#  VQA        - VQA class that loads VQA annotation file and prepares data structures.
#  getQuesIds - Get question ids that satisfy given filter conditions.
#  getImgIds  - Get image ids that satisfy given filter conditions.
#  loadQA     - Load questions and answers with the specified question ids.
#  showQA     - Display the specified questions and answers.
#  loadRes    - Load result file and create result object.

# Help on each function can be accessed by: "help(COCO.function)"

import json
import datetime
import copy
import io


def _load_file(file):
    if isinstance(file, io.IOBase):
        return json.load(file)
    elif isinstance(file, str):
        return json.load(open(file, 'r'))
    else:
        return file


class VQA:
    def __init__(self, annotation_file=None, question_file=None):
        """
           Constructor of VQA helper class for reading and visualizing questions and answers.
        :param annotation_file (str): location of VQA annotation file
        :return:
        """
        # load dataset
        self.dataset = {}
        self.questions = {}
        self.qa = {}
        self.qqa = {}
        self.imgToQA = {}
        if annotation_file is not None and question_file is not None:
            print('loading VQA annotations and questions into memory...')
            time_t = datetime.datetime.utcnow()
            dataset = _load_file(annotation_file)
            questions = _load_file(question_file)
            print(datetime.datetime.utcnow() - time_t)
            self.dataset = dataset
            self.questions = questions
            self.create_index()

    def create_index(self):
        # create index
        print('creating index...')
        img_to_qa = {ann['image_id']: [] for ann in self.dataset['annotations']}
        qa = {ann['question_id']: [] for ann in self.dataset['annotations']}
        qqa = {ann['question_id']: [] for ann in self.dataset['annotations']}
        for ann in self.dataset['annotations']:
            img_to_qa[ann['image_id']] += [ann]
            qa[ann['question_id']] = ann
        for ques in self.questions['questions']:
            qqa[ques['question_id']] = ques
        print('index created!')

        # create class members
        self.qa = qa
        self.qqa = qqa
        self.imgToQA = img_to_qa

    def info(self):
        """
        Print information about the VQA annotation file.
        :return:
        """
        for key, value in self.datset['info'].items():
            print('%s: %s' % (key, value))

    def get_ques_ids(self, img_ids=[], ques_types=[], ans_types=[]):
        """
        Get question ids that satisfy given filter conditions. default skips that filter
        :param: img_ids    (int array)   : get question ids for given imgs
                ques_types (str array)   : get question ids for given question types
                ans_types  (str array)   : get question ids for given answer types
        :return:    ids   (int array)   : integer array of question ids
        """
        img_ids = img_ids if type(img_ids) == list else [img_ids]
        ques_types = ques_types if type(ques_types) == list else [ques_types]
        ans_types = ans_types if type(ans_types) == list else [ans_types]

        if len(img_ids) == len(ques_types) == len(ans_types) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(img_ids) == 0:
                anns = sum([self.imgToQA[imgId] for imgId in img_ids if imgId in self.imgToQA], [])
            else:
                anns = self.dataset['annotations']
            anns = anns if len(ques_types) == 0 else [ann for ann in anns if ann['question_type'] in ques_types]
            anns = anns if len(ans_types) == 0 else [ann for ann in anns if ann['answer_type'] in ans_types]
        ids = [ann['question_id'] for ann in anns]
        return ids

    def get_img_ids(self, ques_ids=[], ques_types=[], ans_types=[]):
        """
        Get image ids that satisfy given filter conditions. default skips that filter
        :param: ques_ids   (int array)   : get image ids for given question ids
                ques_types (str array)   : get image ids for given question types
                ans_types  (str array)   : get image ids for given answer types
        :return: ids     (int array)   : integer array of image ids
        """
        ques_ids = ques_ids if type(ques_ids) == list else [ques_ids]
        ques_types = ques_types if type(ques_types) == list else [ques_types]
        ans_types = ans_types if type(ans_types) == list else [ans_types]

        if len(ques_ids) == len(ques_types) == len(ans_types) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(ques_ids) == 0:
                anns = sum([self.qa[quesId] for quesId in ques_ids if quesId in self.qa], [])
            else:
                anns = self.dataset['annotations']
            anns = anns if len(ques_types) == 0 else [ann for ann in anns if ann['question_type'] in ques_types]
            anns = anns if len(ans_types) == 0 else [ann for ann in anns if ann['answer_type'] in ans_types]
        ids = [ann['image_id'] for ann in anns]
        return ids

    def load_qa(self, ids=[]):
        """
        Load questions and answers with the specified question ids.
        :param ids (int array)       : integer ids specifying question ids
        :return: qa (object array)   : loaded qa objects
        """
        if type(ids) == list:
            return [self.qa[id] for id in ids]
        elif type(ids) == int:
            return [self.qa[ids]]

    def show_qa(self, anns):
        """
        Display the specified annotations.
        :param anns (array of object): annotations to display
        :return: None
        """
        if len(anns) == 0:
            return 0
        for ann in anns:
            quesId = ann['question_id']
            print("Question: %s" % (self.qqa[quesId]['question']))
            for ans in ann['answers']:
                print("Answer %d: %s" % (ans['answer_id'], ans['answer']))

    def load_res(self, res_file, ques_file):
        """
        Load result file and return a result object.
        :param   res_file (str)     : file name of result file
        :return: res (obj)         : result api object
        """

        res = VQA()
        res.questions = _load_file(ques_file)

        res.dataset['info'] = copy.deepcopy(self.questions['info'])
        res.dataset['task_type'] = copy.deepcopy(self.questions['task_type'])
        res.dataset['data_type'] = copy.deepcopy(self.questions['data_type'])
        res.dataset['data_subtype'] = copy.deepcopy(self.questions['data_subtype'])
        res.dataset['license'] = copy.deepcopy(self.questions['license'])

        print('Loading and preparing results...     ')
        time_t = datetime.datetime.utcnow()

        anns = _load_file(res_file)

        assert type(anns) == list, 'results is not an array of objects'
        anns_ques_ids = [ann['question_id'] for ann in anns]
        print(len(set(anns_ques_ids)), len(set(self.get_ques_ids())))
        # assert set(anns_ques_ids) == set(self.get_ques_ids()), \
        #     'Results do not correspond to current VQA set. Either the results do not have predictions for all ' \
        #     'question ids in annotation file or there is atleast one question id that does not belong to the ' \
        #     'question ids in the annotation file.'
        for ann in anns:
            ques_id = ann['question_id']
            if res.dataset['task_type'] == 'Multiple Choice':
                assert ann['answer'] in self.qqa[ques_id][
                    'multiple_choices'], 'predicted answer is not one of the multiple choices'
            qa_ann = self.qa[ques_id]
            ann['image_id'] = qa_ann['image_id']
            ann['question_type'] = qa_ann['question_type']
            ann['answer_type'] = qa_ann['answer_type']
        print('DONE (t=%0.2fs)' % ((datetime.datetime.utcnow() - time_t).total_seconds()))

        res.dataset['annotations'] = anns
        res.create_index()
        return res
