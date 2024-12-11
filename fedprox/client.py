"""Defines the MNIST Flower Client and a function to instantiate it."""

from collections import OrderedDict
from typing import Callable, Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
from flwr.common.typing import NDArrays, Scalar
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from fedprox.models import test, train
from flwr.common import GetParametersRes, Status, Code, FitIns, FitRes, EvaluateIns, EvaluateRes, ndarrays_to_parameters, parameters_to_ndarrays
import io
is_straggler_flag = True

# pylint: disable=too-many-arguments
class FlowerClient(
    fl.client.NumPyClient
):  # pylint: disable=too-many-instance-attributes
    """Standard Flower client for CNN training."""

    def __init__(
        self,
        net: torch.nn.Module,
        trainloader: DataLoader,
        valloader: DataLoader,
        device: torch.device,
        num_epochs: int,
        learning_rate: float,
        straggler_schedule: np.ndarray,
    ):  # pylint: disable=too-many-arguments
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.straggler_schedule = straggler_schedule

    # def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
    #     """Return the parameters of the current net."""
    #     return [val.cpu().numpy() for _, val in self.net.state_dict().items()]
    def get_parameters(self, config: Dict[str, Scalar]) -> GetParametersRes:
        parameters = [val.cpu().numpy() for _, val in self.net.state_dict().items()]
        parameters_proto = ndarrays_to_parameters(parameters)
        return GetParametersRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=parameters_proto,
        )


    def set_parameters(self, parameters: NDArrays) -> None:
        """Change the parameters of the model using the given ones."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)


    def fit(self, fit_ins: FitIns) -> FitRes:
        # 从fit_ins中提取参数和config
        parameters = parameters_to_ndarrays(fit_ins.parameters)
        config = fit_ins.config
        self.set_parameters(parameters)

        # 判断straggler逻辑
        if (
            self.straggler_schedule[int(config["curr_round"]) - 1]
            and self.num_epochs > 1
        ):
            num_epochs = np.random.randint(1, self.num_epochs)

            if config["drop_client"]:
                # 不进行训练, 返回当前参数与对应状态
                current_parameters_res = self.get_parameters({})
                return FitRes(
                    status=Status(code=Code.OK, message="Dropped client"),
                    parameters=current_parameters_res.parameters, # 直接使用 GetParametersRes 返回的 parameters
                    num_examples=len(self.trainloader),
                    metrics={"is_straggler": True}
                )
        else:
            num_epochs = self.num_epochs

        # 训练逻辑
        train(
            self.net,
            self.trainloader,
            self.device,
            epochs=num_epochs,
            learning_rate=self.learning_rate,
            proximal_mu=float(config["proximal_mu"]),
        )

        # 获取更新后的参数
        updated_parameters_res = self.get_parameters({})

        # 返回FitRes对象
        return FitRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=updated_parameters_res.parameters,
            num_examples=len(self.trainloader),
            metrics={"is_straggler": False}
        )

    def evaluate(self, evaluate_ins: EvaluateIns) -> EvaluateRes:
        parameters = parameters_to_ndarrays(evaluate_ins.parameters)
        config = evaluate_ins.config
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)

        return EvaluateRes(
            status=Status(code=Code.OK, message="Success"),
            loss=float(loss),
            num_examples=len(self.valloader),
            metrics={"accuracy": float(accuracy)},
        )

    # def fit(
    #     self, parameters: NDArrays, config: Dict[str, Scalar]
    # ) -> Tuple[NDArrays, int, Dict]:
    #     """Implement distributed fit function for a given client."""
    #     self.set_parameters(parameters)

    #     # At each round check if the client is a straggler,
    #     # if so, train less epochs (to simulate partial work)
    #     # if the client is told to be dropped (e.g. because not using
    #     # FedProx in the server), the fit method returns without doing
    #     # training.
    #     # This method always returns via the metrics (last argument being
    #     # returned) whether the client is a straggler or not. This info
    #     # is used by strategies other than FedProx to discard the update.
    #     if (
    #         self.straggler_schedule[int(config["curr_round"]) - 1]
    #         and self.num_epochs > 1
    #     ):
    #         num_epochs = np.random.randint(1, self.num_epochs)

    #         if config["drop_client"]:
    #             # return without doing any training.
    #             # The flag in the metric will be used to tell the strategy
    #             # to discard the model upon aggregation
    #             return (
    #                 self.get_parameters({}),
    #                 len(self.trainloader),
    #                 {"is_straggler": True},
    #             )

    #     else:
    #         num_epochs = self.num_epochs

    #     train(
    #         self.net,
    #         self.trainloader,
    #         self.device,
    #         epochs=num_epochs,
    #         learning_rate=self.learning_rate,
    #         proximal_mu=float(config["proximal_mu"]),
    #     )

    #     return self.get_parameters({}), len(self.trainloader), {"is_straggler": False}

    # def evaluate(
    #     self, parameters: NDArrays, config: Dict[str, Scalar]
    # ) -> Tuple[float, int, Dict]:
    #     """Implement distributed evaluation for a given client."""
    #     self.set_parameters(parameters)
    #     loss, accuracy = test(self.net, self.valloader, self.device)
    #     return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


def gen_client_fn(
    num_clients: int,
    num_rounds: int,
    num_epochs: int,
    trainloader: DataLoader,
    valloader: DataLoader,
    learning_rate: float,
    stragglers: float,
    model: DictConfig,
) -> Callable[[str], FlowerClient]:  # pylint: disable=too-many-arguments
    """Generate the client function that creates the Flower Clients.

    Parameters
    ----------
    num_clients : int
        The number of clients present in the setup
    num_rounds: int
        The number of rounds in the experiment. This is used to construct
        the scheduling for stragglers
    num_epochs : int
        The number of local epochs each client should run the training for before
        sending it to the server.
    trainloaders: List[DataLoader]
        A list of DataLoaders, each pointing to the dataset training partition
        belonging to a particular client.
    valloaders: List[DataLoader]
        A list of DataLoaders, each pointing to the dataset validation partition
        belonging to a particular client.
    learning_rate : float
        The learning rate for the SGD  optimizer of clients.
    stragglers : float
        Proportion of stragglers in the clients, between 0 and 1.

    Returns
    -------
    Callable[[str], FlowerClient]
        A client function that creates Flower Clients.
    """
    # Defines a straggling schedule for each clients, i.e at which round will they
    # be a straggler. This is done so at each round the proportion of straggling
    # clients is respected
    stragglers_mat = np.transpose(
        np.random.choice(
            [0, 1], size=(num_rounds, num_clients), p=[1 - stragglers, stragglers]
        )
    )

    def client_fn(cid: str) -> FlowerClient:
        """Create a Flower client representing a single organization."""
        # Load model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = instantiate(model).to(device)
        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data

        return FlowerClient(
            net,
            trainloader,
            valloader,
            device,
            num_epochs,
            learning_rate,
            stragglers_mat[int(cid)],
        )

    return client_fn

from typing import Dict, Union

import flwr as fl
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from fedprox import client, server, utils
from fedprox.dataset import load_client_dataloader

FitConfig = Dict[str, Union[bool, float]]
client_num = 0

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    log_filename = ""

    partition_id = cfg.partition_id
    globals()['client_num'] = partition_id
    print (client_num)
  
    fl.common.logger.configure(identifier="myFlowerExperiment", filename=log_filename)

    trainloader,valloader = load_client_dataloader(client_num)
    print("client",trainloader, valloader)    
    client_fn = client.gen_client_fn(
        num_clients=cfg.num_clients,
        num_epochs=cfg.num_epochs,
        trainloader=trainloader,
        valloader=valloader,
        num_rounds=cfg.num_rounds,
        learning_rate=cfg.learning_rate,
        stragglers=cfg.stragglers_fraction,
        model=cfg.model,
    )
    flower_client = client_fn(str(partition_id))
    fl.client.start_client(server_address="localhost:8080", client=flower_client)

if __name__ == "__main__":
    main()