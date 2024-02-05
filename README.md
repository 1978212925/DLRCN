This repository plays a pivotal role in our research paper titled "The Prognostic Value of CT-Derived Adipose Tissue in Diagnosing Malignancy of Pulmonary Nodules: A Multicenter Study". 
The raw code underlying our manuscript, which has been submitted to Nature Communications.
The code consists of five sections. 
"Utils" includes feature extraction, feature selection and nomogram establishment. 
It is worth nothing that adipose tissue features was saved as csv files, while IPN deep learning features can only be extracted in code, due to dimension problems.
"runner" is used to train the deep learning model, including some necessary hyperparamete"rs.
"models" preserve the deep learning models needed in the experiment.
The function of "image_preprocessing_toolkit", like its name, is to preprocess the image.
The code in "evaluation_visualization" is used to do some simple tests during the training of model.
"evaluation_indicators" contains most of the visualizations in the paper
