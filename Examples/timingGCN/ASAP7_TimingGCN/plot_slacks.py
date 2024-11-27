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

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pdb
matplotlib.use('TkAgg')

for b in ['usbf_device']:
    data = np.load('checkpoints/18_slacksdump/{}.npz'.format(b))
    se, sl, se_truth, sl_truth = [data['arr_{}'.format(i)].flatten() for i in range(4)]

    def plot_slack(ax, se_truth, se, title):
        # pdb.set_trace()
        ax.set_title(title)
        ax.scatter(se_truth, se, s=10)
        ax.set_xlabel('Truth')
        ax.set_ylabel('Predicted')
        # line = np.polyfit(se_truth, se, 1)
        # y1 = line[0] * se_truth + line[1]
        # ax.plot(se_truth, y1, 'r-')
        ax.plot(se_truth, se_truth, 'r-')

    fig, ax = plt.subplots(1, 2)
    # fig.tight_layout()
    plt.subplots_adjust(left=0.125, bottom=0.2, right=0.9, top=0.9, wspace=0.4, hspace=0.2)
    plot_slack(ax[0], se_truth, se, 'Hold slacks')
    ax[0].xaxis.set_ticks([-5, -2.5, 0, 2.5, 5])
    ax[0].yaxis.set_ticks([-5, -2.5, 0, 2.5, 5])
    plot_slack(ax[1], sl_truth, sl, 'Setup slacks')
    ax[1].xaxis.set_ticks([-10, -5, 0, 5, 10])
    ax[1].yaxis.set_ticks([-10, -5, 0, 5, 10])
    fig.canvas.manager.set_window_title(b)
    plt.show()
