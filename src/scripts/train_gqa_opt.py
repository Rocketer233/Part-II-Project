import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from src.datamodule import GLUEDataModule
from model import OPTClassifier
import src.models.opt.convert_checkpoint_opt as convert_checkpoint_opt
import src.models.opt.modeling_opt as modeling_opt
import src.models.opt.modeling_opt_gqa as modeling_opt_gqa
import src.models.llama.modeling_llama as modeling_llama
import src.models.llama.modeling_llama_gqa as modeling_llama_gqa
from opt_grouping import *

import csv

# tensorboard --logdir lightning_logs/ --port 6006

# print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_names = {'OPT': 'facebook/opt-125m', 'LLaMA': 'JackFram/llama-160m'}
tasks = [['mnli', 3]]

model_name = 'facebook/opt-125m'
task_name = 'sst2'
num_labels = 2

res = list()

for task_name, num_labels in tasks:

    model = modeling_opt.OPTForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    config = model.config
    data_module = GLUEDataModule(model_name=model_name, task_name=task_name)
    data_module.setup("fit")

    classifier = OPTClassifier(model)

    trainer = pl.Trainer(max_epochs=3)
    trainer.fit(classifier, data_module)

    for grouping in kv_grouping(model, avg=True):
        model_type = f'OPT-GQA-KVAVG_#Group={len(grouping[0])}-Pooling'

        config.groups_idx = grouping
        
        gqa_model = modeling_opt_gqa.OPTForSequenceClassification(config)
        state = model.state_dict()
        gqa_model.load_state_dict(convert_checkpoint_opt.mha2gqa(state, grouping, num_heads=12, transpose_layer=True))

        classifier = OPTClassifier(gqa_model)

        logger = TensorBoardLogger("lightning_logs", name=task_name + "-" + model_type)
        trainer = pl.Trainer(max_epochs=3, logger=logger)
        trainer.fit(classifier, data_module)

        total_params = sum(p.numel() for p in gqa_model.parameters())

        # print(f"total params: {total_params}")

        total_flops = 0
        for _, module in gqa_model.named_modules():
            if hasattr(module, "flops"):
                total_flops += module.flops
        # print(f"total flops: {total_flops}")
        
        res.append((total_params, total_flops))

    print(res)

    with open("results_kvavg.csv", "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerows(res)
