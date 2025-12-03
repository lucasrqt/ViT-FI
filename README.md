# ViT-FI
Software fault injector to emulate neutron beaum faults on Vision Transformers (ViTs).

## TNS data reproducability

To allow the reproducability of the results, we provide the data gathered during the experiments used to generate the plots.

### Starters

First you have to clone this repository.

```bash
git clone https://github.com/lucasrqt/ViT-FI.git
```

Once cloned, go to the folder containing the different data.

```bash
cd ViT-FI/
```

### Software fault-injections

All the code and parsed results are in `TNS2025/fi_results`. To parse the results and get the plots, run the jupyter notebook `TNS2025_FI_results.ipynb`.

The parsed results are stored in the `parsed_results/` folder.


**N.B.** the folder contains Excel sheets used to plot the figures. The figures were plot with Excel, therefore the generated pdfs differ in terms of style. We provide the graph template used to do the figures in `TNS2025/misc/`.

## How to use the fault injector

**TODO**


## Contact

Don't hesitate to contact me for any question at `lucas.roquet@inria.fr`.