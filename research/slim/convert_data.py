# Copyright (C) 2018  Vadym Gryshchuk (vadym.gryshchuk@protonmail.com)
# Date created: 29 July 2018
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Converts a dataset: CORe50, ICWT or NICO Vision.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from datasets import convert_core50
from datasets import convert_icwt
from datasets import convert_nico

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'dataset_name',
    None,
    'The name of the dataset to convert: core50, icwt or nico.')

tf.app.flags.DEFINE_string(
    'training_dataset_dir',
    None,
    'The folder, which contains the train data.')

tf.app.flags.DEFINE_string(
    'testing_dataset_dir',
    None,
    'The folder, which contains the test data.')

tf.app.flags.DEFINE_string(
    'convert_dataset_dir',
    None,
    'The folder where the TFRecords and class labels are saved.')


def main(_):
    if not FLAGS.testing_dataset_dir:
        raise ValueError('ERROR: provide the test data with --testing_dataset_dir')
    if not FLAGS.training_dataset_dir:
        raise ValueError('ERROR: provide the train data with --training_dataset_dir')
    if not FLAGS.convert_dataset_dir:
        raise ValueError('ERROR: provide output folder with --convert_dataset_dir')
    if not FLAGS.dataset_name:
        raise ValueError('ERROR: provide the dataset name with --dataset_name')

    if FLAGS.dataset_name == 'core50':
        convert_core50.run(FLAGS.training_dataset_dir, FLAGS.testing_dataset_dir,
                                                 FLAGS.convert_dataset_dir)
    elif FLAGS.dataset_name == 'icwt':
        convert_icwt.run(FLAGS.training_dataset_dir, FLAGS.testing_dataset_dir,
                                               FLAGS.convert_dataset_dir)
    elif FLAGS.dataset_name == 'nico':
        convert_nico.run(FLAGS.training_dataset_dir, FLAGS.testing_dataset_dir,
                                               FLAGS.convert_dataset_dir)
    else:
        raise ValueError(
            'The name of the dataset [%s] was not recognized.' % FLAGS.dataset_name)


if __name__ == '__main__':
    tf.app.run()
