# Evolutionary Dynamic Optimization Laboratory (EDOLAB)

> A MATLAB Optimization Platform for Education and Experimentation in Dynamic Environments

**Git repository:** [EvoMindLab/EDOLAB](https://github.com/EvoMindLab/EDOLAB)

**The current version is `v2.10`**

## 📰 News

- **[December 2025]** 🎉 This work has been published in **ACM Transactions on Mathematical Software** (JCR Q1 - CORE/ERA A* Ranked Journal)!
  - **Paper**: [M. Peng, D. Yazdani, D. Yazdani, Z. She, W. Luo, C. Li, J. Branke, T. T. Nguyen, A. H. Gandomi, S. Yang, Y. Jin, and X. Yao, "Algorithm 1060: EDOLAB, a Platform for Research and Education in Evolutionary Dynamic Optimization," *ACM Transactions on Mathematical Software*, 2025.](https://dl.acm.org/doi/10.1145/3785134)  
  - **DOI**: [10.1145/3785134](https://doi.org/10.1145/3785134)
  - **Citation**: If you use EDOLAB in your research, please cite:
    ```bibtex
    @article{mai2026EDOLAB,
    author = {Peng, Mai and Yazdani, Delaram and Yazdani, Danial and She, Zeneng and Luo, Wenjian and Li, Changhe and Branke, Juergen and Nguyen, Trung Thanh and Gandomi, Amir H. and Yang, Shengxiang and Jin, Yaochu and Yao, Xin},
    title = {Algorithm 1060: EDOLAB, a Platform for Research and Education in Evolutionary Dynamic Optimization},
    journal = {ACM Transactions on Mathematical Software},
    publisher = {Association for Computing Machinery},
    url = {https://doi.org/10.1145/3785134},
    doi = {10.1145/3785134},
    articleno = {4},
    year = {2026},
    volume = {52},
    number = {1},
    pages={1--30}
    }
    ```

## Get Started

```bash
# clone the project
git clone https://github.com/EvoMindLab/EDOLAB.git
```

## Architecture Overview
Modular structure with core components:
```
├── Algorithm/             # EDOA implementations (ACFPSO, CDE, mQSO, etc.)
├── Benchmark/             # Benchmark generators (GMPB, MPB, FPs, GDBG)
├── Utility/               # Helper functions
├── OctaveVersion/         # Octave supports
├── PythonVersion/         # Python version (pip-installable)
├── CEC2027_Competition/   # CEC 2027 competition edition (runs on PythonVersion)
├── GUIMode.m              # GUI entry point
└── CodeMode.m             # Script mode entry
```

## Running EDOLAB

> The current version of the GUI Mode is relatively well-developed, and we strongly recommend using it.


### GUI Mode (MATLAB R2020b+)
```matlab
% Launch graphical interface
run GUIMode.m
```

* **Experimentation module** 

**Core Features:**
- Experiment configuration management
- Batch task processing
- Real-time progress monitoring
- Statistical significance analysis
- Interactive visualizations (Trend/Box plots)

![GUI Demo](https://github.com/PengMai1998/EDOLAB-Assets/blob/main/ExperimentationModule.png?raw=true)

* **Education module** 

**Core Features:**
- Step-by-step visualization of optimization process
- Interactive control over iteration playback
- Real-time population behavior tracking
- Interactive control, freely start, stop, or jump to any iteration

![GUI Demo](https://github.com/PengMai1998/EDOLAB-Assets/blob/main/EducationModule.png?raw=true)


### Code Mode (MATLAB R2016b+/Octave)

1. **Select algorithm and benchmark in `CodeMode.m`**

   ```matlab
   % Configure Algorithm & Benchmark in CodeMode.m
   AlgorithmName = 'mQSO';
   BenchmarkName = 'MPB';
   VisualizationOverOptimization = 0;  % 1 = Education Mode
   ```

2. **Where configurable parameters are defined**

   Each algorithm and benchmark exposes its configurable parameters through
   dedicated functions:

   - Algorithm parameters:  
     `Algorithm/<AlgorithmName>/getAlgConfigurableParameters_<AlgorithmName>.m`
   - Benchmark parameters:  
     `Benchmark/<BenchmarkName>/getProConfigurableParameters_<BenchmarkName>.m`

   Inside these files, parameters follow a uniform pattern:

   ```matlab
   ConfigurableParameters.ParamName = struct( ...
       'value', <default>, ...
       'type',  '<numeric|integer|option|boolean>', ...
       'range', [...], ...
       'description', '...' );
   ```

   You can discover all available parameter names programmatically in MATLAB:

   ```matlab
   tmpAlg = getAlgConfigurableParameters(AlgorithmName);
   fieldnames(tmpAlg)           % Algorithm parameter names

   tmpPro = getProConfigurableParameters(BenchmarkName);
   fieldnames(tmpPro)           % Benchmark parameter names
   ```

3. **Override parameters directly in `CodeMode.m` (recommended for fast experiments)**

   In `CodeMode.m`, parameters are first loaded from the files above:

   ```matlab
   ConfigurableAlgParameters = getAlgConfigurableParameters(AlgorithmName);
   ConfigurableProParameters = getProConfigurableParameters(BenchmarkName);
   ```

   You can then override any parameter for the current run only by changing
   the `.value` fields:

   ```matlab
   % Algorithm-side overrides (example for ACFPSO)
   ConfigurableAlgParameters.PopulationSize.value = 20;
   ConfigurableAlgParameters.c1.value             = 1.8;
   ConfigurableAlgParameters.c2.value             = 1.6;

   % Benchmark-side overrides (example for GMPB / MPB)
   ConfigurableProParameters.Dimension.value        = 10;
   ConfigurableProParameters.ChangeFrequency.value  = 2000;
   ConfigurableProParameters.PeakNumber.value       = 5;
   ```

   These overrides do **not** change any JSON or GUI configuration; they only
   affect the current execution of `CodeMode.m`.

4. **Run Code Mode**

   In MATLAB or Octave, simply execute:

   ```matlab
   run CodeMode.m
   ```

## Extension

### Adding New Algorithms
1. **Create algorithm folder**  
   Under `Algorithm/`, create a new sub-folder named after your EDOA (e.g. `XYZ`).

2. **Implement the five core functions**  
   Place these files inside your new folder:
   - `main_XYZ.m`  
     Main entry point—must accept and return the standardized Problem and Output structures.
   - `SubPopulationGenerator_XYZ.m`  
     Generates or updates the algorithm’s subpopulations.
   - `IterativeComponents_XYZ.m`  
     Contains the core search/update loop for each iteration.
   - `ChangeReaction_XYZ.m`  
     Handles detection of and adaptation to environmental changes.
   - `getAlgConfigurableParameters_XYZ.m`  
     Defines all user-configurable parameters (fields: `value`, `type`, `range`, `description`).

3. **Hook into CodeMode**  
   Ensure that `CodeMode.m` can invoke `main_XYZ.m` by setting:
   ```matlab
   AlgorithmName = 'XYZ';
   ```

4. **Visualization & Output**

   * Use the `%% Visualization for education` section pattern from existing EDOAs to collect GUI data.
   * At the end of `main_XYZ.m`, assemble an `Results.Indicators` structure consistent with EDOLAB’s reporting modules.


### Adding New Benchmarks

1. **Create benchmark folder**
   Under `Benchmark/`, create a sub-folder named after your benchmark (e.g. `ABC`).

2. **Implement the three core files**
   Inside `Benchmark/ABC/` add:

   * `getProConfigurableParameters_ABC.m`
     Define `ConfigurableParameters` with fields:

     ```matlab
     ConfigurableParameters.ParamName = struct( ...
       'value', <default>, ...
       'type',  '<numeric|integer|option>', ...
       'range', [...], ...
       'description', '…' ...
     );
     ```
   * `BenchmarkGenerator_ABC.m`
     Builds the `Problem` structure (environments, dynamics) based solely on the parameters from `getProConfigurableParameters_ABC.m`.
   * `fitness_ABC.m`
     Computes the baseline fitness for each solution, conforming to EDOLAB’s input/output conventions.

3. **Automatic integration**
   Once these files are in place, your new benchmark will appear in the GUI list and can be run via `CodeMode.m`.


### Custom Performance Metrics

1. **Edit the JSON interface**
   Open `Indicators/indicators.json` and add:

   ```json
   "MyMetric": {
     "type": "FE based"          // or "Environment based" / "None"
   }
   ```

2. **Implement in `fitness.m`**
   In `fitness.m`, compute and store your metric:

   ```matlab
   % FE based (every evaluation)
   Problem.Indicators.MyMetric.trend(Problem.FE) = <your calculation>;

   % Environment based (per environment)
   Problem.Indicators.MyMetric.trend(Problem.EnvironmentCounter) = <your calculation>;

   % None (final only)
   Problem.Indicators.MyMetric.final = <your calculation>;
   ```

3. **Verify & Use**

   * In **GUI Mode**, run one task and check the Completed Tasks panlel or Statistical Analysis panel to see your new indicator.
   * In **Code Mode**, confirm `Results.Indicators.MyMetric` appears in results.


## Octave Support
EDOLAB provides full functionality in Octave through its dedicated OctaveVersion implementation. This standalone version maintains all core features while addressing key compatibility requirements.

1. Install required packages:
```octave
pkg install -forge io statistics
pkg load io statistics
```
2. Compile KDTree component:
```bash
cd OctaveVersion/Benchmark
mkoctfile --mex ConstructKDTree.cpp
```

3. Execute the main controller script:
```bash
run OctaveVersion/OctaveCodeMode.m
```


## Python Version
`PythonVersion/` is an object-oriented Python implementation of EDOLAB. It covers the benchmarks GMPB, MPB, FPs,
GDBG and DynPORTAL, published EDOAs, the offline error and best-before-change indicators, batch experiments with
parallel runs, and statistical analysis (Friedman with Nemenyi, Wilcoxon). It requires Python 3.10 or later.

```bash
cd PythonVersion
python -m pip install -e .
python -m cli list
python -m cli run --algorithm mQSO --problem GMPB --runs 31 --problem-param Dimension=5
```

It can also be installed without cloning:

```bash
python -m pip install "git+https://github.com/EvoMindLab/EDOLAB.git#subdirectory=PythonVersion"
```

See [PythonVersion/README.md](PythonVersion/README.md) for batch experiments, statistical analysis and adding
algorithms.


## CEC 2027 Competition
`CEC2027_Competition/` holds the 24 problems F1–F12 (GMPB) and P1–P12 (DynPORTAL) of the CEC 2027 Competition on
Dynamic Optimization Problems, a runner that writes the files of a submission, an integrity check of the test problems,
two example algorithms and a template. It runs on `PythonVersion/`. **The test problems must not be modified.**

```bash
python -m pip install -e PythonVersion
python -m CEC2027_Competition.run --algorithm Example_mQSO --smoke
```

See [CEC2027_Competition/README.md](CEC2027_Competition/README.md).


## For More Information

For more information about EDOLAB, please refer to the [paper](https://dl.acm.org/doi/10.1145/3785134) and user manual. If you need further assistance, please contact Mai Peng at [pengmai1998@gmail.com](mailto:pengmai1998@gmail.com) or Danial Yazdani at [danial.yazdani@gmail.com](mailto:danial.yazdani@gmail.com).

## MATLAB Support
<img src="https://www.mathworks.com/etc.clientlibs/mathworks/clientlibs/customer-ui/templates/common/resources/images/pic-header-mathworks-logo.20221030234646672.svg" width="35%">
<br>

* **GUI Mode** requires [MATLAB R2020b+](https://www.mathworks.com/products/new_products/release2020b.html) and the **Parallel Computing Toolbox**. Recommended [MATLAB R2024a](https://www.mathworks.com/products/new_products/release2024a.html) or later.
* **Code Mode** requires [MATLAB R2016b+](https://www.mathworks.com/help/doc-archives.html).


## License

This program is to be used under the terms of the [GNU](http://www.gnu.org/copyleft/gpl.html) General Public License

Copyright (c) 2022-present **Danial Yazdani**
