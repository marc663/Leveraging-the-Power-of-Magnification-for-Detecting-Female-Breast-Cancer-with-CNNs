# Leveraging-the-Power-of-Magnification-for-Detecting-Female-Breast-Cancer-with-CNNs

The aim of this project is to apply deep learning techniques to whole slide images of histological lymph node sections. Metastases in lymph nodes is a deciding factor to prognosticate breast cancer.The diagnostic procedure of examining lymph nodes requires extensive microscopic assessmentby pathologists and small metastases are difficult to detect. An early detection of breast canceris vital. Patients who have been diagnosed with breast cancer at Stage 1 have significantly higher survival rates, than patients who have been diagnosed at Stage 4. From this we can deduce that the earlier the cancer is detected, the higher the chances of survival for the patient. Studies have shown that deep learning is capable of detecting and classifying characterising diseases on a cellular level in medical images and can even outperform medical professionals. Automated solutions thus reduce the workload of pathologists and reduce the subjectivity in diagnosis. The research in algorithms for automated detection and classification of breast cancer metastases in whole-slide images is therefore an essential part of the treatment of breast cancer. In this project will be using a a slightly adapted Inception-ResNet-v1 model in order to detect breast cancer metastases in whole slide images of histological lymph node sections.

This project used data from the CAMELYON data set https://camelyon17.grand-challenge.org/.

The project has the following structure:
1) Generating annotation masks
2) Patch extraction
3) Stain Normalization
4) Loading the data and traiing the model
5) Testing
