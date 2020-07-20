# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch

from fairseq import utils

from .label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion
from . import register_criterion


@register_criterion('tsa_label_smoothed_cross_entropy')
class LabelSmoothedCrossEntropyCriterionWithTSA(LabelSmoothedCrossEntropyCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        sample['target'] = sample['target'][:, 1:].contiguous()
        net_output = model.compute(**sample['net_input'])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def accuracy(output, target):
        with torch.no_grad():
            _, pred = output.topk(1, -1)
            correct = pred.view(-1).eq(target.view(-1))
        return correct.sum()
