import torch
import torch.nn as nn
import csv
import os
import json
import logging
from fp16 import FP16_Module
import GPUtil
from collections import OrderedDict
from settings import args, MODEL_CLASS, TOKENIZER, SPECIAL_TOKEN_IDS, init_logging
from settings import MEMORY_FACTOR, LEN_FACTOR, TASK_DICT, MODEL_CONFIG, DATA_ATTRS, SPECIAL_TOKENS, CONFIG_CLASS, CONFIG_NAME
from utils import QADataset, top_k_top_p_filtering, create_dataloader, logits_to_tokens, get_model_dir
from utils import sample_sequence, remove_id, get_gen_token, lll_unbound_setting
from metrics import compute_metrics, normalize_text
logger = logging.getLogger(__name__)


def SVD(A, idd, tpos, THD=0.99):
    U, Sigma, V = torch.svd(A)
    sum_singu = Sigma.sum()
    singus = 0.
    cnt = 0
    for sigma in Sigma:
        singus += sigma
        cnt += 1
        if (singus / sum_singu) > THD:
            break
    logger.info('The rank of token_pos ' + str(tpos) + ' layer ' + str(idd) + ' is: ' + str(cnt))
    return V[:, 0:cnt], Sigma[0:cnt], cnt

def test_one_to_one(task_load, task_eval, model, score_dict, s_tokens):

    logger.info("start to test { task: %s (load) %s (eval), seq train type: %s }" % (task_load, task_eval, args.seq_train_type))

    test_qadata = QADataset(TASK_DICT[task_eval]["train"] , "train", SPECIAL_TOKEN_IDS[task_load]).sort()
    max_a_len = test_qadata.max_a_len
    test_dataloader = create_dataloader(test_qadata, "test")
    n_examples = len(test_qadata)
    logger.info("len of test dataset: {}".format(n_examples))

    need_process = OrderedDict()
    qa_results = [0 for _ in range(n_examples)]
    all_pasts = [[0 for _ in range(n_examples)] for __ in range(MODEL_CONFIG.n_layer)]
    max_tot_lens = [0 for _ in range(n_examples)]

    cnt = 0
    tot_loss = 0.
    loss_fct = nn.CrossEntropyLoss(reduction='sum')
    save_hidden_states = [[torch.zeros(0) for _ in range(MODEL_CONFIG.n_layer)] for i in range(args.control_len)]
    count_examples = 0
    for n_steps, (cqs, len_cqs, _, _, Y, _, _) in enumerate(test_dataloader):
        now_cqs = cqs[0]
        len_cqs = len_cqs[0]
        n_inputs = now_cqs.shape[0]
        count_examples += n_inputs
        logger.info("done {}, total {}, now {}%".format(count_examples, n_examples, count_examples*100.0/n_examples))

        past = s_tokens.unsqueeze(0).expand(n_inputs, -1, -1).cuda()
        bsz, seqlen, _ = past.shape
        past = past.view(bsz, seqlen, MODEL_CONFIG.n_layer * 2,
                         MODEL_CONFIG.n_head, MODEL_CONFIG.n_embd // MODEL_CONFIG.n_head)#.type(torch.half)
        past = past.permute([2, 0, 3, 1, 4]).split(2)
        all_outputs = model(input_ids=now_cqs.cuda(), past=past)
        del past
        torch.cuda.empty_cache()
        
        outputs = all_outputs[0]
        if "gpt2" in args.model_name:
            pasts = all_outputs[1]
            all_hidden_states = all_outputs[2]
            for ii in range(len(all_hidden_states)-1):
                for jj in range(args.control_len):
                    save_hidden_states[jj][ii] = torch.cat([save_hidden_states[jj][ii], all_hidden_states[ii][range(n_inputs), len_cqs-1-jj, :].cpu()])
            del all_hidden_states

        next_logits = outputs[range(n_inputs), len_cqs-1, :] / args.temperature_qa
        tot_loss = tot_loss + loss_fct(next_logits, Y[0][range(n_inputs),len_cqs-1].cuda()).item()

        next_tokens = logits_to_tokens(next_logits).cpu()

        for i in range(n_inputs):
            max_tot_lens[cnt] = max_a_len + test_qadata[cnt][1]
            qa_results[cnt] = now_cqs[i][:len_cqs[i]]
            if next_tokens[i] != SPECIAL_TOKEN_IDS["eos_token"]:
                qa_results[cnt] = torch.cat((now_cqs[i][:len_cqs[i]], next_tokens[i]))
                if len(qa_results[cnt]) not in [max_tot_lens[cnt], args.max_len]:
                    need_process.update([[cnt, None]])
                    if "gpt2" in args.model_name:
                        for layer_id in range(MODEL_CONFIG.n_layer):
                            all_pasts[layer_id][cnt] = pasts[layer_id][:, i, ..., :len_cqs[i], :].type(torch.float)#torch.float if args.fp32 else torch.half)#.type(torch.float32).cpu()# if args.fp32 else torch.half)
            cnt += 1
       
        if len(need_process) > int(1 * args.memory_sizes[0] / now_cqs.shape[1]):  # dynamic threshold to avoid out of memory
            sample_sequence(model, need_process, qa_results, all_pasts, max_tot_lens)

        del now_cqs
        torch.cuda.empty_cache()

    sample_sequence(model, need_process, qa_results, all_pasts, max_tot_lens)

    sel = []
    for i in range(len(test_qadata)):
        _, len_cq, _, _, Y, _, _, _ = test_qadata[i]
        Y = list(filter(lambda x: x != -1, Y))[:-1]  # remove eos
        Y = ' '.join([str(y) for y in Y]).split(str(SPECIAL_TOKEN_IDS["pad_token"]))
        Y = [TOKENIZER.decode(list(map(int, y.split()))) for y in Y]
        now_ans = TOKENIZER.decode(qa_results[i].tolist()[len_cq:])
        qa_results[i] = [now_ans, Y]
        sel.append(normalize_text(now_ans, task_eval) == normalize_text(Y[0], task_eval))

    sel = torch.tensor(sel)
    for jj in range(args.control_len):
        save_hidden_states[jj] = [_[sel] for _ in save_hidden_states[jj]]

    print(save_hidden_states[0][0].shape)

    Proj = [[] for jj in range(args.control_len)]
    for jj in range(args.control_len):
        for i in range(MODEL_CONFIG.n_layer):
            G = save_hidden_states[jj][i] - save_hidden_states[jj][i].mean(dim=0, keepdim=True)
            Princ_axis, _, _ = SVD(G, i, jj)
            Proj[jj].append(torch.mm(Princ_axis, Princ_axis.t()))

    torch.save(Proj, os.path.join(model.model_dir, 'Proj_{}_p{}_ep{}_cl{}.pth.tar'.format(task_eval,args.preseqlen,model.ep+1,args.control_len)))

def test_one_to_many(task_load):
    score_dicts = []
    ep = args.test_ep - 1
    model_dir = get_model_dir([task_load])
    s_tokens_path = os.path.join(model_dir, "p"+str(args.preseqlen)+'lr'+str(args.learning_rate)+'model-stokens{}'.format(ep+1))
    config_path = os.path.join(model_dir,CONFIG_NAME)

    gen_token = get_gen_token(task_load)
    TOKENIZER.add_tokens([gen_token])
    SPECIAL_TOKENS[task_load] = gen_token
    SPECIAL_TOKEN_IDS[task_load] = TOKENIZER.convert_tokens_to_ids(gen_token)
    model = MODEL_CLASS.from_pretrained('../gpt2-medium-pretrained/').cuda()
    model.resize_token_embeddings(len(TOKENIZER))
    model.transformer.output_hidden_states = True
    model.transformer.output_attentions = False

    s_tokens = torch.load(s_tokens_path).cpu().to("cuda")#.cuda()
    
    model.ep = ep
    model.model_dir = model_dir
    score_dict = {k:None for k in args.tasks}
    logger.info("task: {}, epoch: {}".format(task_load, ep+1))
    with torch.no_grad():
        for task_eval in args.tasks:
            test_one_to_one(task_load, task_eval, model, score_dict, s_tokens)

if __name__ == '__main__':
    if args.n_gpus > 1:
        raise NotImplementedError("test can be run with only one gpu currently!")
    
    if not args.debug:
        logging.getLogger("pytorch_transformers").setLevel(logging.WARNING)
        logging.getLogger("pytorch_transformers.tokenization_utils").setLevel(logging.CRITICAL)
    init_logging(os.path.join(args.model_dir_root, 'log_test_p{}_lr{}_get_Proj.txt'.format(args.preseqlen, args.learning_rate)))
    logger.info('args = {}'.format(args))

    for task_load in args.tasks:
        test_one_to_many(task_load)
