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
import random

random.seed(8026728)

available_data = 'blabla usb_cdc_core BM64 jpeg_encoder salsa20 usbf_device aes128 wbqspiflash aes192 cic_decimator xtea aes256 des spm y_huff aes_cipher picorv32a synth_ram zipdiv genericfir usb'.split()

data = {k: list(map(lambda t: t.to('cuda'), torch.load('data/4_netstat/{}.netstat.pt'.format(k))))
        for k in available_data}

for k, (x, y) in data.items():
    y = torch.log(0.0001 + y) + 7.6
    data[k] = x, y

example_d = next(iter(data.values()))
num_input_features = example_d[0].shape[1]
num_outputs = example_d[1].shape[1]

train_data_keys = random.sample(available_data, 14)
data_train = {k: g for k, g in data.items() if k in train_data_keys}
data_test = {k: g for k, g in data.items() if k not in train_data_keys}

if __name__ == '__main__':
    print('Netstat total {} benchmarks'.format(len(data)))
