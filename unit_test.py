import unittest
from data_utils import *
import os
import json
import pprint


class TestDataUtilsMethods(unittest.TestCase):
    def test_get_dialogs(self):
        print(get_dialogs("data/dialog-bAbI-tasks/dialog-babi-task1-API-calls-trn.txt")[0])

    def test_load_dialog_task(self):
        train_data, test_data, val_data = load_dialog_task("data/dialog-bAbI-tasks", 1)
        print(len(train_data))
        print(len(test_data))
        print(len(val_data))

    def test_load_candidates(self):
        candidates, d = load_candidates("data/dialog-bAbI-tasks", 1)
        print(d['api_call italian rome six cheap'])

    def test_dstc2_data(self):
        train_dev_path = 'data/dstc2_traindev'
        test_path = 'data/dstc2_test'
        def get_flist(data_path, dtype='train'):
            flist_path = os.path.join(data_path, 'scripts/config/dstc2_' + dtype + '.flist')
            with open(flist_path) as file:
                data = file.read().split('\n')
            return data[:-1]
        train_flist = get_flist(train_dev_path, 'train')
        dev_flist = get_flist(train_dev_path, 'dev')
        test_flist = get_flist(test_path, 'test')

        data_root_path = os.path.join(train_dev_path, 'data')
        slot_dict = {}
        for file in train_flist:
            json_file_path = os.path.join(data_root_path, file, 'log.json')
            with open(json_file_path) as json_file:
                data = json.load(json_file)
                turns = data['turns']
                for turn in turns:
                    in_put = turn['input']
                    out_put = turn['output']
                    if len(in_put['live']['slu-hyps']) is 0:
                        input_slots = []
                    else:
                        input_slots = in_put['live']['slu-hyps'][0]['slu-hyp']

                    output_slots = out_put['dialog-acts']
                    for slot in input_slots + output_slots:
                        slot = slot['slots']
                        if len(slot) == 0:
                            continue
                        slot = slot[0]
                        value = slot_dict.get(slot[0], set())
                        value.add(slot[1])
                        slot_dict[slot[0]] = value

        print(len(slot_dict['name']))
        pprint.pprint(slot_dict)


if __name__ == '__main__':
    unittest.main()