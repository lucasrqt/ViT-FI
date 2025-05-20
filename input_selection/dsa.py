from methods import InputSelection, InputSelectionFactory
from utils.input_selection import InputSelectionMethod, DSAUtils, DSAHook
import configs
import torch
import os


@InputSelectionFactory.register(InputSelectionMethod.DSA)
class DSA(InputSelection):
    """
    DSA input selection method.

    Adapted from the code of the paper "Guiding Deep Learning System Testing using Surprise Adequacy" (https://doi.org/10.1109/ICSE.2019.00108)
    source code available at: https://github.com/coinse/sadl/blob/master/sa.py
    """

    def __init__(
        self, train_loader, val_loader, model, model_name, save_path, device=configs.CPU, min_batch: int = 0, max_batch: int = 0
    ):
        super().__init__(method=InputSelectionMethod.DSA)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.model_name = model_name
        self.device = device
        self.hook = None
        self.handler = None
        self.save_path = save_path
        self.min_batch = min_batch
        self.max_batch = max_batch

    def __get_hook(self) -> torch.utils.hooks.RemovableHandle:
        """
        Get the hook for the model.
        """
        handlers = DSAUtils.get_hookable_layers(self.model, layer=configs.MLP)
        hook, handler = DSAUtils.select_layer(handlers, len(handlers) - 1)
        return hook, handler

    def __get_target_ats(self):
        """
        Get the at of .
        """
        raise NotImplementedError("Not implemented yet.")

    def __get_trained_ats(self):
        """
        Get the trained activation traces of the training dataset.
        """
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.train_loader):
                if i < self.min_batch:
                    continue
                print(i)

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                _ = self.model(inputs)
                # Assuming the model has a method to get attention scores
                ats = self.hook.get_ats()
                classes, counts = torch.unique(labels, return_counts=True)
                offset = 0
                for cls_label, count in zip(classes, counts):
                    if count > 0:
                        ats_for_cls = ats[offset : offset + count]
                        filename = DSAUtils.construct_file_name(
                            self.model_name,
                            configs.FP32,
                            self.hook.layer_name,
                            cls_label.item(),
                        )
                        if not DSAUtils.is_ats_present(self.save_path, filename):
                            DSAUtils.save_ats(
                                ats_for_cls.cpu(), self.save_path, filename
                            )
                        else:
                            ats_to_cat = DSAUtils.load_ats(self.save_path, filename)
                            ats_to_cat = torch.cat(
                                (ats_to_cat.cpu(), ats_for_cls.cpu()), dim=0
                            )
                            DSAUtils.save_ats(ats_to_cat, self.save_path, filename)
                        offset += count.item()
                self.hook.clear_ats()

                if i == (self.max_batch - 1):
                    break

    def __fetch_dsa(self):
        """
        Fetch the DSA scores of the model.
        """
        hook, handler = self.__get_hook()
        self.hook = hook
        self.handler = handler

        self.__get_trained_ats()

    def find_closest_at(self):
        """
        Find the closest attention scores to the trained target attention scores.
        """
        raise NotImplementedError("Not implemented yet.")

    def select_input(self):
        DSAUtils.check_folder(self.save_path)
        self.__fetch_dsa()
