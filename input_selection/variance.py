import torch
import configs
from methods import InputSelection, InputSelectionFactory
from utils.input_selection import InputSelectionMethod
from utils import result_data_utils


@InputSelectionFactory.register(InputSelectionMethod.VARIANCE)
class Variance(InputSelection):
    """
    Variance input selection method.
    """

    def __init__(
        self,
        model,
        dataloader,
        number_of_classes,
        df_res,
        k=configs.K,
        device=configs.CPU,
    ):
        super().__init__(method=InputSelectionMethod.VARIANCE)
        self.model = model
        self.dataloader = dataloader
        self.df_res = df_res
        self.k = k
        self.device = device
        self.number_of_classes = number_of_classes

    def __enable_dropout(self):
        """
        Enable dropout for the model.
        """
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.train()

    # TODO: manage case where last batch is not full
    def select_input(self):
        """
        Select inputs based on the variance of the model outputs.

        Formula: Var(input) = 1/C \times \sum_{i=1}^{C} var(P_i(input))
        where C is the number of classes
        P_i(input) = {p_{i}^{j}|0<j<=k}, where k is the number of passes
        and p_{i}^{j} is the output of the model for the j-th pass of class i
        and input is the input image.
        """
        self.__enable_dropout()

        # gathering outputs
        for inputs, _ in self.dataloader:
            outputs = [None] * self.k
            for i in range(self.k):
                inputs = inputs.to(self.device)
                with torch.no_grad():
                    outputs[i] = self.model(inputs)

            # computing variance
            outputs = torch.stack(outputs, dim=0)

            for j in range(len(inputs)):
                var = torch.var(outputs[:, j, :], dim=0).mean().item()
                self.df_res = result_data_utils.input_selection_append_row(
                    self.df_res, var
                )
