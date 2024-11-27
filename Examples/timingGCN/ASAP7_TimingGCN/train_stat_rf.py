# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# SPDX-License-Identifier: Apache-2.0

#

# Licensed under the Apache License, Version 2.0 (the "License");

# you may not use this file except in compliance with the License.

# You may obtain a copy of the License at

#

# http://www.apache.org/licenses/LICENSE-2.0

#

# Unless required by applicable law or agreed to in writing, software

# distributed under the License is distributed on an "AS IS" BASIS,

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# See the License for the specific language governing permissions and

# limitations under the License.

import numpy as np
from sklearn.ensemble import RandomForestRegressor
import random
import pickle
import pdb

from data_stat_cpu import data_train, data_test, num_input_features, num_outputs

from sklearn.metrics import r2_score


def test(model):
    print('======= Training dataset ======')
    for k, (x, y) in data_train.items():
        print(k, r2_score(model.predict(x), y))
    print('======= Test dataset ======')
    for k, (x, y) in data_test.items():
        print(k, r2_score(model.predict(x), y))

def train(model):
    model = RandomForestRegressor(verbose=1, n_jobs=48)
    x, y = data_train_ensemble
    model.fit(x, y)
    with open('./checkpoints/netstat_rf.pickle', 'wb') as f:
        pickle.dump(model, f)

if __name__ == '__main__':
    with open('./checkpoints/netstat_rf.pickle', 'rb') as f:
        model = pickle.load(f)
    model.verbose = 0
    test(model)
    # train(model)
