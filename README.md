# An-ensemble-learning-for-microplastcs-identification-with-FTIR

Microplastics (MPs) (size < 5 mm) marine pollution have been investigated and monitored by many researchers and found in many coasts around the world. These toxic chemicals make their way into human diet through food chain when aquatic organisms ingest MPs. Attenuated Total Reflection Fourier transform infrared spectroscopy (ATR-FTIR) is a very effective method to detect MPs. To provide the automatic detecting method for MPs, Numerous studies have proposed Machine Learning (ML) based methods, such as Support Vector Machines, K-Nearest Neighbours, and Random Forests, for identification and classification of MPs through using the ATR-FTIR data. The evaluations of these ML based methods primarily focus on the average scores across all types of MPs. However, the existing FTIR datasets are normally imbalanced. Furthermore, some MPs contain the identical functional group, and some MPs may be fouled or contaminated, which will reduce the quality of FTIR data samples (e.g. lacking of peaks or creating noises). These factors will interfere the ML classification algorithms and cause the algorithms to perform differently while identifying different MPs. Hence, this work proposes an ensemble learning algorithm to exploit the advantage of different ML algorithms based on a systematic evaluation of the existing ML based MP identification approaches. A neural network is employed to fuse the outputs of chosen ML algorithms to improve the overall metrics. The evaluation results show that the proposed algorithm outperforms existing single ML based approaches.


## # Datasets





## # Citation

Use the below bibtex to cite us.

```BibTeX
@article{yan2022ensemble,
  title={An ensemble machine learning method for microplastics identification with FTIR spectrum},
  author={Yan, Xinyu and Cao, Zhi and Murphy, Alan and Qiao, Yuansong},
  journal={Journal of Environmental Chemical Engineering},
  volume={10},
  number={4},
  pages={108130},
  year={2022},
  publisher={Elsevier}
}

@misc{yan2023ensemble,
  title={An ensemble machine learning method for microplastics identification with FTIR spectrums},
  author={Yan, Xinyu and Cao, Zhi and Murphy, Alan and Qiao, Yuansong},
  year={2023},
  publisher={Github},
  howpublished={\url{https://github.com/lyheiyu/An-ensemble-learning-for-microplastcs-identification-with-FTIR/}},
}

```
* * * * *

## Developed by

[Software Research Institute](https://sri.ait.ie/) of [Technological University of the Shannon: Midlands Midwest](https://tus.ie/).

