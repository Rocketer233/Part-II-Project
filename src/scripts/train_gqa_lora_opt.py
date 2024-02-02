import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from src.datamodule import GLUEDataModule
from model import OPTClassifier
from src.models.opt.configuration_opt_lora import OPTLoraConfig
import src.models.opt.convert_checkpoint_opt as convert_checkpoint_opt
import src.models.opt.modeling_opt_lora as modeling_opt_lora
import src.models.opt.modeling_opt_gqa_lora as modeling_opt_gqa_lora
from lora_utils import (
    print_trainable_parameters,
    mark_only_lora_as_trainable
)
from opt_grouping import get_neighbour_groups
import toml

# tensorboard --logdir lightning_logs/ --port 6006

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tasks = [['sst2', 2]]

model_name = 'facebook/opt-125m'
config_files = ["lora_by_type.toml"]


for config_file in config_files:
    # load toml config file
    with open(config_file, "r") as f:
        lora_config = toml.load(f)
    print(f"LoRA PEFT with {config_file} config file successfully loaded!")

# lora_config is a dict representation of the toml file

for task_name, num_labels in tasks:
    for group_size in (1, 2, 4, 8):
        model_type = f'OPT-GQA-LoRA-GroupSize={group_size}-Pooling'
        
        peft_config = OPTLoraConfig.from_pretrained(model_name, lora_config=lora_config, num_labels=num_labels)
        model = modeling_opt_lora.OPTForSequenceClassification.\
                from_pretrained(model_name, config=peft_config)

        groups_idx = get_neighbour_groups(group_size=group_size)
        print(groups_idx)
        model.config.groups_idx = groups_idx
        gqa_model = modeling_opt_gqa_lora.OPTForSequenceClassification(model.config)
        state = model.state_dict()
        gqa_model.load_state_dict(convert_checkpoint_opt.mha2gqa_lora(state, groups_idx, num_heads=12, transpose_layer=True))
        gqa_model = mark_only_lora_as_trainable(gqa_model)
        print_trainable_parameters(gqa_model)


        data_module = GLUEDataModule(model_name=model_name, task_name=task_name)
        data_module.setup("fit")

        classifier = OPTClassifier(gqa_model)

        logger = TensorBoardLogger("lightning_logs", name=task_name + "-" + model_type)
        trainer = pl.Trainer(max_epochs=30, logger=logger)
        trainer.fit(classifier, data_module)