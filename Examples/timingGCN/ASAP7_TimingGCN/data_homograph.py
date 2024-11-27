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

import torch

def gen_homograph():
    from data_graph import data_train, data_test, gen_homobigraph_with_features
    # replace hetero graph with homographs
    # do not execute this in other modules, as it would modify
    # the global data in a dirty way
    for dic in [data_train, data_test]:
        for k in dic:
            g, ts = dic[k]
            dic[k] = gen_homobigraph_with_features(g)

    torch.save([data_train, data_test], './data/7_homotest/train_test.pt')

data_train, data_test = torch.load('./data/7_homotest/train_test.pt')
    
if __name__ == '__main__':
    # gen_homograph()
    pass

