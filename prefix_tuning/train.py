import torch
from torch import nn
from pytorch_transformers import AdamW
import os
import logging
from fp16 import FP16_Optimizer
from parallel import DataParallelCriterion
from utils import *
from settings import args, TASK_DICT, init_logging, MODEL_CONFIG, MODEL_CLASS, SPECIAL_TOKENS
from settings import TOKENIZER, SPECIAL_TOKEN_IDS, FILL_VAL, SAVE_NAME, TOKENS_WEIGHT, CONFIG_NAME
from scheduler import AnnealingLR
from torch.nn import CrossEntropyLoss
logger = logging.getLogger(__name__)


def train(task_ids, model):
    tasks = [args.tasks[task_id] for task_id in task_ids]

    logger.info("start to train { task: %s, seq train type: %s }" % (tasks, args.seq_train_type))
    model_dir = get_model_dir(tasks)
    make_dir(model_dir)

    train_dataset = [TASK_DICT[t]["train"] for t in tasks]
    train_extra_data = []
    logger.info('extra training data size: {}'.format(len(train_extra_data)))

    if not model:
        # model = MODEL_CLASS.from_pretrained('../gpt2-medium-pretrained/').cuda()
        model = MODEL_CLASS.from_pretrained('gpt2-medium').cuda()
        model.resize_token_embeddings(len(TOKENIZER))
        model.train()
        for p in model.parameters(): p.requires_grad = False
    print(model.transformer.wte.weight[0,:].shape)
    print(model.transformer.wte.weight[0,:])

    prefix_tokens = torch.arange(args.preseqlen).long()
    prefix_weight = nn.Embedding(args.preseqlen, MODEL_CONFIG.n_embd).requires_grad_(True).to(args.device_ids[0])
    prefix_weight.from_pretrained(model.transformer.wte.weight[:args.preseqlen,:])

    control_trans = nn.Sequential(
                        nn.Linear(MODEL_CONFIG.n_embd, args.mid_dim), #1024 * 512
                        nn.Tanh(),
                        nn.Linear(args.mid_dim, MODEL_CONFIG.n_layer * 2 * MODEL_CONFIG.n_embd)).requires_grad_(True).to(args.device_ids[0])
    print(prefix_weight.weight.shape)
    print(control_trans[0].weight.shape, control_trans[2].weight.shape)

    gen_token = get_gen_token(tasks[0])
    TOKENIZER.add_tokens([gen_token])
    TOKENIZER.save_pretrained(model_dir)
    SPECIAL_TOKENS[tasks[0]] = gen_token
    SPECIAL_TOKEN_IDS[tasks[0]] = TOKENIZER.convert_tokens_to_ids(gen_token)
    logger.info('gen token = {} , gen token id = {}'.format(gen_token, SPECIAL_TOKEN_IDS[tasks[0]]))
    MODEL_CONFIG.vocab_size = len(TOKENIZER)
    MODEL_CONFIG.to_json_file(os.path.join(model_dir,CONFIG_NAME))
    global TOKENS_WEIGHT
    if len(TOKENIZER) != TOKENS_WEIGHT.shape[0]:
        TOKENS_WEIGHT = torch.cat((TOKENS_WEIGHT, torch.ones([1]).cuda()))

    model.resize_token_embeddings(len(TOKENIZER))
    for p in model.parameters(): p.requires_grad = False

    model = WrapModel(model)
#    model = DataParallelModel(WrapModel(model), args.device_ids)

    train_qadata = QADataset(train_dataset, "train", SPECIAL_TOKEN_IDS[tasks[0]], train_extra_data)
    max_train_batch_size = max(len(train_qadata) // args.min_n_steps, args.min_batch_size)
    train_dataloader = create_dataloader(train_qadata, "train", max_train_batch_size)
    n_train_epochs = args.n_train_epochs[tasks[0]]
    n_train_optimization_steps = len(train_qadata) * n_train_epochs
    logger.info('len of train dataset: {} , max train batch size {} , num of opt steps: {}'.format(
        len(train_qadata), max_train_batch_size, n_train_optimization_steps))

    param_optimizer = [prefix_weight.weight, control_trans[0].weight, control_trans[2].weight]#list(filter(lambda p: p.requires_grad, model.parameters()))#list(model.named_parameters())
    print([param_optimizer[i].shape for i in range(len(param_optimizer))])
    
    optimizer = AdamW(param_optimizer, lr=args.learning_rate, eps=args.adam_epsilon)
    if not args.fp32:
        optimizer = FP16_Optimizer(optimizer, static_loss_scale=None, dynamic_loss_scale=True,
                                   dynamic_loss_args={'scale_window': 100, 'min_scale': 1, 'delayed_shift': 2})

    scheduler = AnnealingLR(optimizer, start_lr=args.learning_rate, warmup_iter=int(args.n_warmup_ratio*len(train_qadata)),
            num_iters=int(n_train_optimization_steps), decay_style=args.decay_style)
    train_loss_fct = DataParallelCriterion(CrossEntropyLoss(ignore_index=FILL_VAL, weight=TOKENS_WEIGHT), args.device_ids)

    tot_n_steps = 0
    train_once = TrainStep(model, optimizer, scheduler)

    for ep in range(n_train_epochs):
        cum_loss, cum_qa_loss, cum_lm_loss, cur_n_inputs = 0, 0, 0, 0
        for n_steps, (_, _, cqa, _, Y, gen_X, gen_Y) in enumerate(train_dataloader):

            n_inputs = cqa[0].shape[0]
            prefix_tokensi = prefix_tokens.unsqueeze(0).expand(cqa[0].shape[0], -1).to(args.device_ids[0])
            temp_control   = prefix_weight(prefix_tokensi)#.to(args.device_ids[0]) # preseqlen, emb (20*1024)
            past           = control_trans(temp_control)#.to(args.device_ids[0])  #bsz, preseqlen, layer*emb
            bsz, seqlen, _ = past.shape
            past           = past.view(bsz, seqlen, MODEL_CONFIG.n_layer * 2, 
                                       MODEL_CONFIG.n_head, MODEL_CONFIG.n_embd // MODEL_CONFIG.n_head)
            past           = past.permute([2, 0, 3, 1, 4]).split(2)

            cqa_ = cqa[0].to(args.device_ids[0])
            Y_ = Y[0].to(args.device_ids[0])
            gen_X_ = gen_X[0].to(args.device_ids[0])
            gen_Y_ = gen_Y[0].to(args.device_ids[0])

            losses = get_losses(model, cqa_, Y_, gen_X_, gen_Y_, train_loss_fct, past) #parallel_model
            loss = sum(losses)
            train_once(loss, n_inputs)

            qa_loss = losses[0].item() * n_inputs
            lm_loss = losses[1].item() * n_inputs
            cum_loss += (qa_loss + lm_loss)
            cum_qa_loss += qa_loss
            cum_lm_loss += lm_loss
            cur_n_inputs += n_inputs

            if (n_steps + 1) % args.logging_steps == 0:
                logger.info('progress {:.3f} , lr {:.1E} , loss {:.3f} , qa loss {:.3f} , lm loss {:.3f} , avg batch size {:.1f}'.format(
                    ep + cur_n_inputs/len(train_qadata), scheduler.get_lr(), cum_loss/cur_n_inputs, cum_qa_loss/cur_n_inputs, cum_lm_loss/cur_n_inputs,
                    cur_n_inputs/(n_steps + 1)
                ))

        torch.save(control_trans(prefix_weight(prefix_tokens.to(prefix_weight.weight.device))).cpu(), os.path.join(model_dir, "p"+str(args.preseqlen)+"lr"+str(args.learning_rate)+SAVE_NAME+"stokens"+str(ep+1)))
        tot_n_steps += (n_steps + 1)
        logger.info('epoch {}/{} done , tot steps {} , lr {:.1E} , loss {:.2f} , qa loss {:.2f} , lm loss {:.2f} , avg batch size {:.1f}'.format(
            ep+1, n_train_epochs, tot_n_steps, scheduler.get_lr(), cum_loss/cur_n_inputs, cum_qa_loss/cur_n_inputs, cum_lm_loss/cur_n_inputs, cur_n_inputs/(n_steps+1)
        ))

    return model


if __name__ == '__main__':

    if not args.debug:
        logging.getLogger("pytorch_transformers").setLevel(logging.WARNING)
        logging.getLogger("pytorch_transformers.tokenization_utils").setLevel(logging.CRITICAL)

    make_dir(args.model_dir_root)

    init_logging(os.path.join(args.model_dir_root, 'log_train_p{}_lr{}.txt'.format(args.preseqlen, args.learning_rate)))
    logger.info('args = {}'.format(str(args)))

    model = None
    for task_id in range(len(args.tasks)):
        model = train([task_id], model)
