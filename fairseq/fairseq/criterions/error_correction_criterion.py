import math
import numpy as np

import torch
import torch.nn.functional as F

from fairseq import utils
from . import FairseqCriterion, register_criterion
from .label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion, label_smoothed_nll_loss


@register_criterion('error_correction_criterion')
class ErrorCorrectionCriterion(LabelSmoothedCrossEntropyCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        
        use_cuda = torch.cuda.is_available() and not self.args.cpu
        self.step = 0
        self.t    = 5000
        self.accum = args.update_freq[0]

        
        self.alpha = args.alpha
        self.beta = args.beta
        self.mask_idx = task.target_dictionary.add_symbol('<mask>')
        self.smooth = args.smooth
        
    @staticmethod
    def add_args(parser):
        LabelSmoothedCrossEntropyCriterion.add_args(parser)
        parser.add_argument('--teacher-path',
                            metavar='FILE',
                            help="path(s) to teacher model file(s) colon separated")
        parser.add_argument('--alpha', 
                            default=0.85,
                            type=float,
                            help="coefficient for query loss")
        parser.add_argument('--beta', 
                            default=1.0,
                            type=float,
                            help="coefficient for query loss")
        parser.add_argument('--smooth',
                            default=0.01,
                            type=float)


    def forward(self, model, sample, reduce=True):
        self.step += 1
        sample['target'] = sample['target'][:, 1:].contiguous()

        if model.training is False:
            inputs = self.prepare_input(sample, model, eval=True)
        else:
            inputs = self.prepare_input(sample, model)
        net_output = model.compute(**inputs)
        
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        acc = utils.item(self.accuracy(net_output[0], sample['target']))

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
            'acc': acc,
        }

        positive_logits = net_output[1][inputs['return_content_mask']]
        negative_logits = net_output[1][~inputs['return_content_mask']]
        positive_targets = sample['target'][inputs['return_content_mask']]
        negative_targets = sample['target'][~inputs['return_content_mask']]
        positive_loss, _ = self.compute_electra_loss(positive_logits, positive_targets)
        negative_loss, _ = self.compute_electra_loss(negative_logits, negative_targets)

        logging_output['c_loss'] = utils.item(positive_loss.data)
        logging_output['c_size'] = positive_targets.numel()
        logging_output['c_acc'] = utils.item(self.accuracy(positive_logits, positive_targets))

        loss = loss + positive_loss * self.beta + negative_loss * self.smooth

        return loss, sample_size, logging_output

    def compute_sample_probs(self):
        step = self.step // self.accum
        t = self.t
        ### 
        if step <= 30000:
            return 1.0
        ###

        return max(self.alpha, (t / (t + math.exp(step / t))))
    
    def compute_electra_loss(self, logits, targets):
        lprobs = utils.log_softmax(logits, dim=-1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, targets, self.eps, ignore_index=self.padding_idx, reduce=True,
        )
        return loss, nll_loss

    def prepare_input(self, sample, model, eval=False, k=5):
        sz = sample['target'].size()
        with torch.no_grad():
            is_pad = sample['target'].eq(1)
            p = self.compute_sample_probs()
            if eval is False and False:
                choose = (torch.rand(is_pad.size()) > p).to(is_pad)
                choose = (~choose | is_pad).long()
                #model.eval()
                net_output = model.compute(**sample['net_input'])
                _, pred = net_output[0].topk(k, dim=-1)
                t = (torch.from_numpy(np.random.choice(k, sz[0] * sz[1])) + torch.arange(0, sz[0] * sz[1] * k, k)).to(sample['target'])
                topk = pred.reshape(-1)[t].reshape(sz)
                new_input = topk * (1 - choose) + sample['target'] * choose
                masked_tokens = (1 - choose).bool()
                #model.train()
            else:
                new_input = sample['target']
                masked_tokens = new_input != sample['target']

            new_input = torch.cat((sample['net_input']['prev_output_tokens'][:, :1], new_input), dim=-1)

            return {
                'src_tokens': sample['net_input']['src_tokens'],
                'src_lengths': sample['net_input']['src_lengths'],
                'prev_output_tokens': new_input, 
                'return_content_mask': masked_tokens,
            }

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        c_size = sum(log.get('c_size', 0) for log in logging_outputs)
        c_loss = sum(log.get('c_loss', 0) for log in logging_outputs)
        c_acc  = sum(log.get('c_acc', 0) for log in logging_outputs)

        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2) if sample_size > 0 else 0.,
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2) if ntokens > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
            'acc': sum(log.get('acc', 0) for log in logging_outputs) / ntokens,
            'c_loss': c_loss / c_size / math.log(2) if c_size != 0 else 0,
            'c_acc': c_acc / c_size if c_size != 0 else 0,
        }

    @staticmethod
    def accuracy(output, target):
        with torch.no_grad():
            _, pred = output.topk(1, -1)
            correct = pred.view(-1).eq(target.view(-1))
        return correct.sum()
