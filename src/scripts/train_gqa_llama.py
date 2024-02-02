import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from src.datamodule import GLUEDataModule
from model import OPTClassifier
import src.models.opt.convert_checkpoint_opt as convert_checkpoint_opt
import src.models.llama.convert_checkpoint_llama as convert_checkpoint_llama
import src.models.opt.modeling_opt as modeling_opt
import src.models.opt.modeling_opt_gqa as modeling_opt_gqa
import src.models.llama.modeling_llama as modeling_llama
import src.models.llama.modeling_llama_gqa as modeling_llama_gqa
from opt_grouping import get_neighbour_groups

# tensorboard --logdir lightning_logs/ --port 6006

# print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_names = {'OPT': 'facebook/opt-125m', 'LLaMA': 'JackFram/llama-160m'}
tasks = [['qnli', 2], ['mnli', 3]]

model_name = 'JackFram/llama-160m'

for task_name, num_labels in tasks:
    for group_size in (2, 4, 8):
        model_type = f'LLaMA-GQA-GroupSize={group_size}-Pooling'
        groups_idx = get_neighbour_groups(group_size=group_size)
        model = modeling_llama.LlamaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        print(f"group size = {group_size}")
        print(groups_idx)
        model.config.groups_idx = groups_idx
        gqa_model = modeling_llama_gqa.LlamaForSequenceClassification(model.config)
        state = model.state_dict()
        # for key in state.keys():
        #     print(key, state[key].shape)
        gqa_model.load_state_dict(convert_checkpoint_llama.mha2gqa(state, groups_idx, num_heads=12, transpose_layer=True))


        data_module = GLUEDataModule(model_name=model_name, task_name=task_name)
        data_module.setup("fit")

        classifier = OPTClassifier(gqa_model)

        logger = TensorBoardLogger("lightning_logs", name=task_name + "-" + model_type)
        trainer = pl.Trainer(max_epochs=3, logger=logger)
        trainer.fit(classifier, data_module)