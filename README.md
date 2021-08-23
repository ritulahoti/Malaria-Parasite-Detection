# Malaria-Parasite-Detection
This project aims to automate malaria screening using computer aided diagnosis methods that includes machine learning (ML) and/or Convolutional Neural Network (CNN) techniques, applied to microscopic images of the smears.

The block diagram of the experiment is shown below.

![image](https://user-images.githubusercontent.com/65898464/130516712-2ecbdc57-dd37-46c9-8de3-f9e618a3fced.png)

I) Pre-processing
Texture is a feature used to partition images into ROI and classify them.
• Textural findings could improve the analysis for better diagnostic interpretation.
• It provides information in the spatial arrangement of intensities in an image.
• Segmentation (Morphological operation, K means clustering) is performed to distinguish ROI from the non ROI portion.
• The images are converted to grayscale images, necessary for computing GLCM matrix (Gray Level Co occurrence matrix).

II) Feature Extraction
• Gray Level Co occurrence Matrix (GLCM) is comput ed. Statistical measures are calculated from GLCM matrix such as contrast, entropy, correlation, homogeneity, dissimilarity used as features for further processing.
• Also, local binary pattern (LBP) is used for textural analysis based on differences Also, local binary pattern (LBP) is used for textural analysis based on differences of neighbouring pixels. Gabor filter are also used for extracting features at different hbouring pixels. Gabor filter are also used for extracting features at different orientation and scales (spatialorientation and scales (spatial--frequency localisation). LBP’s and Gabor’s energy frequency localisation). LBP’s and Gabor’s energy and entropy are also computed. The significance of these features are described in and entropy are also computed.
• Correlation between the features are computed and features with high correlation value are retained i.e. homogeneity, energy, lbp entropy, gabor energy, gabor entropy.

![image](https://user-images.githubusercontent.com/65898464/130517327-ea9e5ba5-263d-4b64-8448-b729cfb5b27f.png)

The features so obtained of the training dataset are fed to a learning algorithm which is then used to classify the testing data as parasitized or uninfected.

III) Model Building
The training data was splitted into training (75%) and testing data (25%) forvalidation purpose.
Algorithms experimented: Naive Bayes, Logistic Regression, Support Vector Machine (SVM), Random Forest Classifier
Among all the algorithm for classifying the images as parasitized or uninfected, Random Forest classification was found to predict with better accuracy.

IV) Results

![image](https://user-images.githubusercontent.com/65898464/130517785-75838525-1c4b-4391-a4cd-fc2fff4064e3.png)


