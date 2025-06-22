import hydra
from omegaconf import DictConfig


hydra.main(config_path="configs", config_name="data_eng", version_base=None)
def main(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO)
    data = GetData().get_data(cfg)  
