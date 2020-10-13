# Cifar-10-all-categories
## Purpose
用keras建立一個CNN Model，訓練keras所提供Cifar-10的dataset，
內部共有10種圖片類別，共有60000張，其中50000張為training set，剩餘的10000張為test set。
然後在網路上去下載跟這10種圖片類別相關的圖片，但是並不含在Cifar-10的dataset裡面，利用所訓練好的CNN Model去分類這些圖片分別是哪些種類。
## Define
   * main.py: main function for training and testing
   * picture:從網路上找到10張不是在Cifar-10資料集的32 * 32的圖片，測試看看結果如何
   * 可以在網路上下載32 * 32，且原本不含在Cifar10資料集裡面的圖片，餵進已訓練好的Model中，看看accuracy如何
   * cifar-10.h5: The CNN model
   * model_summary: The structure of the CNN model 
