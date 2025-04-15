import argparse
import configs
import statistical_fi


class MainParser:
    def parse_args(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description="Perform high-level fault injections on ViT model according neutron beam fault model.",
            add_help=True,
        )
        parser.add_argument(
            "-m",
            "--model",
            type=str,
            default=configs.VIT_BASE_PATCH16_224,
            help="Model name.",
            choices=configs.VIT_CLASSIFICATION_CONFIGS,
        )
        parser.add_argument(
            "-D",
            "--dataset",
            type=str,
            default=configs.IMAGENET,
            help="Dataset name.",
            choices=[configs.IMAGENET, configs.COCO, configs.CIFAR10],
        )
        parser.add_argument(
            "-b",
            "--batch-size",
            type=int,
            default=configs.DEFAULT_BATCH_SIZE,
            help="Batch size.",
        )
        parser.add_argument(
            "-p",
            "--precision",
            type=str,
            default=configs.FP32,
            help="Precision of the model and inputs.",
            choices=[configs.FP16, configs.FP32],
        )
        parser.add_argument(
            "-d",
            "--device",
            type=str,
            default=configs.GPU_DEVICE,
            help="Device to run the model.",
            choices=[
                configs.CPU,
                configs.GPU_DEVICE,
                configs.GPU_DEVICE1,
                configs.GPU_DEVICE2,
                configs.GPU_DEVICE3,
            ],
        )
        parser.add_argument(
            "-M",
            "--microop",
            type=str,
            default=None,
            help="Microoperation to inject the fault.",
            choices=configs.MICROBENCHMARK_MODULES,
        )
        parser.add_argument(
            "--target-layer",
            type=lambda lc: statistical_fi.LayerChoice[lc],
            default=statistical_fi.LayerChoice.LAST,
            help="Target layer for the fault injection.",
            choices=list(statistical_fi.LayerChoice),
        )
        parser.add_argument(
            "--injection-type",
            type=lambda it: statistical_fi.InjectionType[it],
            default=statistical_fi.InjectionType.RANDOM,
            help="Type of injection to perform.",
            choices=list(statistical_fi.InjectionType),
        )
        parser.add_argument(
            "-s", "--seed", type=int, default=configs.SEED, help="Random seed."
        )
        parser.add_argument(
            "-S",
            "--shuffle-dataset",
            default=False,
            action="store_true",
            help="Shuffle the dataset or not.",
        )
        parser.add_argument(
            "--fault-model-threshold",
            type=float,
            default=1e-03,
            help="Threshold for the fault model data.",
        )
        parser.add_argument(
            "--inject-on-correct-predictions",
            action="store_true",
            help="Inject faults only on correct predictions.",
            default=False,
        )
        parser.add_argument(
            "--load-critical",
            action="store_true",
            help="Only load the images that are critical for the fault injection.",
            default=False,
        )
        parser.add_argument(
            "--save-critical-logits",
            action="store_true",
            help="Save the logits of the critical images.",
            default=False,
        )
        parser.add_argument(
            "--save-top5prob",
            action="store_true",
            help="Save the top 5 probabilities of the critical images.",
            default=False,
        )
        parser.add_argument(
            "-v", "--verbose", action="store_true", help="Verbose mode.", default=False
        )
        return parser.parse_args()


class InputSelectionParser:
    def parse_args(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description="Generate input selection for DNN testing.",
            add_help=True,
        )

        return parser.parse_args()
