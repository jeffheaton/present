# Washington University [Olin School of Business](https://olin.wustl.edu/EN-US/Pages/default.aspx)
[Center for Analytics and Business Insights](https://olin.wustl.edu/EN-US/Faculty-Research/research-centers/center-analytics-business-insights/Pages/default.aspx) (CABI)  
[Deep Learning for Demand Forecasting](https://github.com/jeffheaton/present/tree/master/WUSTL/CABI-Demand)  
Copyright 2022 by [Jeff Heaton](https://www.youtube.com/c/HeatonResearch), Released under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)  

For this course, we will execute the code using the free version of [Google CoLab](https://colab.research.google.com/), which provides a free [GPU](https://developer.nvidia.com/cuda-gpus). You do not need to install any software on your local computer. The code presented during the course has no dependencies on CoLab, and you could execute the code on your local machine; however, installing deep learning with full GPU support can be a tricky installation process. Also, not all computers have a GPU compatible with deep learning.

You will store your files on [Google Drive](https://www.google.com/drive/). Please make sure that you can access both CoLab and Google Drive. The following video presents a quick overview of Python on CoLab in the way we will use both in this course.

[Introduction to Python and Google CoLab for Data Science and Machine Learning](https://www.youtube.com/watch?v=pNyZUrOQSrE&ab_channel=JeffHeaton)

* [Day 1 Slides](https://github.com/jeffheaton/present/blob/master/WUSTL/CABI-Demand/cabi-dl-demand-forcasting-1.pptx)

Source code for the course: 

* Day 1: Forecasting Demand from Tabular Time Series Data
    * Utility: [GPU Status](https://github.com/jeffheaton/present/blob/master/WUSTL/CABI-Demand/status.ipynb)
    * Example: [Exploratory Data Analysis (EDA)](https://github.com/jeffheaton/present/blob/master/WUSTL/CABI-Demand/demand_eda.ipynb) 
        * Supplemental: [Pandas](https://github.com/jeffheaton/present/blob/master/WUSTL/CABI-Demand/pandas.ipynb), [Pandas Course](https://github.com/jeffheaton/t81_558_deep_learning/blob/master/t81_558_class_02_1_python_pandas.ipynb)
        * [Lab 1: EDA](https://github.com/jeffheaton/present/blob/master/WUSTL/CABI-Demand/lab_1_eda.ipynb), [Lab 1 Solution](https://github.com/jeffheaton/present/blob/master/WUSTL/CABI-Demand/lab_1_eda_solution.ipynb)
    * Example: [Naive Forecast](https://github.com/jeffheaton/present/blob/master/WUSTL/CABI-Demand/demand_naive.ipynb)
        * Supplemental: [Neural Network Overview](https://github.com/jeffheaton/t81_558_deep_learning/blob/master/t81_558_class_03_1_neural_net.ipynb)
        * Supplemental: [Kaggle: Store Item Demand Forecasting Challenge](https://www.kaggle.com/c/demand-forecasting-kernels-only)
    * Example: [Seasonality Aware Forecast](https://github.com/jeffheaton/present/blob/master/WUSTL/CABI-Demand/demand_seasonality.ipynb)
        * [Lab 2: Feature Engineering](https://github.com/jeffheaton/present/blob/master/WUSTL/CABI-Demand/lab-2-features.ipynb), [Lab 2 Solution](https://github.com/jeffheaton/present/blob/master/WUSTL/CABI-Demand/lab_2_features_solution.ipynb)
* Day 2: Forecasting Demand from Unstructured Data
    * Example: [Natural Language Processing Forecast](https://github.com/jeffheaton/present/blob/master/WUSTL/CABI-Demand/demand_nlp.ipynb)
        * Supplemental: [What are Embedding Layers in Keras](https://github.com/jeffheaton/t81_558_deep_learning/blob/master/t81_558_class_11_05_embedding.ipynb)
        * Supplemental: [Word2Vec Demo](https://turbomaze.github.io/word2vecjson/)
    * Example: [Computer Vision Forecast](https://github.com/jeffheaton/present/blob/master/WUSTL/CABI-Demand/demand_cv.ipynb)
        * Supplemental: [Deep Learning in your browser](https://cs.stanford.edu/people/karpathy/convnetjs/)
        * Supplemental: [YOLO](https://pytorch.org/hub/ultralytics_yolov5/)
        * Supplemental: [YOLO Ted Talk](https://www.youtube.com/watch?v=Cgxsv1riJhI&ab_channel=TED)
        * [Lab 3: Computer Vision](https://github.com/jeffheaton/present/blob/master/WUSTL/CABI-Demand/lab-3-cv.ipynb)
    * Example: [Meta Prophet](https://github.com/jeffheaton/present/blob/master/WUSTL/CABI-Demand/demand_prophet.ipynb)
        * Supplemental: [Meta Prophet Official Site](https://facebook.github.io/prophet/)
        * [Lab 4: Forecasting with Prophet](https://github.com/jeffheaton/present/blob/master/WUSTL/CABI-Demand/lab-4-prophet.ipynb)