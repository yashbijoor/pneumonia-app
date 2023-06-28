
# Explaining Autonomous Pneumonia Disease Detection using Grad-CAM for Improved Decision Support

## Abstract

In recent years, artificial intelligence and machine learning has made major strides in the field of medicine. The medical sector requires a high level of accountability and transparency. Explanations for machine decisions and predictions are thus needed to justify their reliability. This requires greater interpretability, which often means it is crucial to understand the mechanism underlying the algorithms. The blackbox nature of deep learning, however, is still unresolved, and many machine decisions are still poorly understood. The reason radiologists are weary of using AI is because they do not trust model predictions without any form of explainability. Thus, we aim to create a system that not only focuses on interpretability and explainability but also has a high enough accuracy to make it reliable enough to be trusted and used by radiologists.

## Confusion matrix

<img width="347" alt="image" src="https://github.com/yashbijoor/pneumonia-app/assets/80248111/599b6c07-70c3-4447-99f4-6e2fe35b9a64">

## Result

![image](https://github.com/yashbijoor/pneumonia-app/assets/80248111/919a3e13-4e4f-4dba-819b-c0a4e6323708)
![image](https://github.com/yashbijoor/pneumonia-app/assets/80248111/07214546-ce34-4ee1-9833-f7f9e5daebba)


## Authors

- [@yashbijoor](https://www.github.com/yashbijoor)
- [@Chahat256](https://www.github.com/Chahat256)
- [@29OmChavan](https://www.github.com/29OmChavan)

## Deployment

Unzip the models

### To deploy the server

```bash
  cd model
```
Install the packages
```bash
  pip install -r requirements.txt
```
Run the server
```bash
  flask run
```
### To deploy the interface

```bash
  cd app
```
Install the packages
```bash
  npm i
```
Run the react app
```bash
  npm start
```

## References

[1] P. Bertin and V. Frappier, “Chester: A web delivered locally computed chest X-ray disease prediction system,” arXiv.org, [https://doi.org/10.48550/arXiv.1901.11210](https://doi.org/10.48550/arXiv.1901.11210)

[2] P. Rajpurkar et al., “Deep learning for chest radiograph diagnosis: A retrospective comparison of the CheXNeXt algorithm to practicing radiologists,” PLOS Medicine, vol. 15, no. 11, 2018. doi:10.1371/journal.pmed.1002686 

[3] A. Das & P. Rad,  “Opportunities and Challenges in Explainable Artificial Intelligence (XAI): A Survey,” 2020

​​[4] E. Tjoa and C. Guan, “A survey on Explainable Artificial Intelligence (XAI): Toward medical xai,” IEEE Transactions on Neural Networks and Learning Systems, vol. 32, no. 11, pp. 4793–4813, 2021. doi:10.1109/tnnls.2020.3027314

​​[5] F. Xu et al., “Explainable AI: A brief survey on history, research areas, approaches and challenges,” Natural Language Processing and Chinese Computing, pp. 563–574, 2019. doi:10.1007/978-3-030-32236-6_51 

[6] S.-H. Lo and Y. Yin, “A novel interaction-based methodology towards explainable AI with better understanding of pneumonia chest X-ray images,” Discover Artificial Intelligence, vol. 1, no. 1, 2021. doi:10.1007/s44163-021-00015-z

[7] B. H. M. van der Velden, H. J. Kuijf, K. G. A. Gilhuijs, and M. A. Viergever, “Explainable artificial intelligence (XAI) in deep learning-based medical image analysis,” Medical Image Analysis, vol. 79, p. 102470, 2022. doi:10.1016/j.media.2022.102470 

[8] M. Ribeiro, S. Singh, and C. Guestrin, “‘Why should i trust you?’: Explaining the predictions of any classifier,” Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Demonstrations, 2016. doi:10.18653/v1/n16-3020 

[9] R. Winastwan, “Interpreting image classification model with lime,” Medium, [https://towardsdatascience.com/interpreting-image-classification-model-with-lime-1e7064a2f2e5](https://towardsdatascience.com/interpreting-image-classification-model-with-lime-1e7064a2f2e5)

[10] F. López, “Shap: Shapley additive explanations,” Medium, [https://towardsdatascience.com/shap-shapley-additive-explanations-5a2a271ed9c3](https://towardsdatascience.com/shap-shapley-additive-explanations-5a2a271ed9c3) (accessed May 19, 2023).

[11] R. R. Selvaraju et al., “Grad-cam: Visual explanations from deep networks via gradient-based localization,” International Journal of Computer Vision, vol. 128, no. 2, pp. 336–359, 2019. doi:10.1007/s11263-019-01228-7 

[12] H. Panwar et al., “A deep learning and grad-cam based color visualization approach for fast detection of COVID-19 cases using chest X-ray and CT-scan images,” Chaos, Solitons & Fractals, vol. 140, p. 110190, 2020. doi:10.1016/j.chaos.2020.110190 

[13] P. Patel, “Chest X-ray (covid-19 & pneumonia),” Kaggle, https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia (accessed May 19, 2023). 

[14] Evaluating the performance of the lime and grad-cam explanation methods on a LEGO multi-label image classification task, [https://www.researchgate.net/publication/343441887_Evaluating_the_performance_of_the_LIME_and_Grad-CAM_explanation_methods_on_a_LEGO_multi-label_image_classification_task](https://www.researchgate.net/publication/343441887_Evaluating_the_performance_of_the_LIME_and_Grad-CAM_explanation_methods_on_a_LEGO_multi-label_image_classification_task)
