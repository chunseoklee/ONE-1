#!/usr/bin/env python3

# Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import h5py
import numpy as np

data_json_path='./sample1.json';
with open(data_json_path) as json_file:
    data = json.load(json_file)


with h5py.File("input_nlu.h5", 'w') as hf:
    h5dtypes = {
        "float32": ">f4",
        "uint8": "u1",
        "bool": "u1",
        "int32": "int32",
        "int64": "int64"
    }
    name_grp = hf.create_group("name")
    val_grp = hf.create_group("value")
    for idx, t in enumerate(data):
        print("idx:" + str(idx) + " t:" + t )
        if t == 'gazet_vecs':
            print("processing gazet_vecs .....")
            val_grp.create_dataset(
                '2', data=data[t], dtype=h5dtypes['float32'])
            name_grp.attrs['2'] = str(t)
        elif t == 'last_label_id':
            print("processing last_label_id .....")
            val_grp.create_dataset(
                '1', data=data[t], dtype=h5dtypes['int32'])
            name_grp.attrs['1'] = str(t)
        elif t == 'word_ids':
            print("processing word_ids .....")
            val_grp.create_dataset(
                '0', data=data[t], dtype=h5dtypes['int32'])
            name_grp.attrs['0'] = str(t)
        elif t == 'reference_capsule_probs':
            print("processing reference_capsule_probs .....")
            d = np.array(data[t], ndmin=2)
            val_grp.create_dataset(
                'expected', data=d, dtype=h5dtypes['float32'])
            name_grp.attrs['expected'] = str(t)
