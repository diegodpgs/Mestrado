import time
import sys
import os
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import nltk
import numpy as np
from inspect import currentframe, getframeinfo
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix


"""
  @author: Diego Pedro
  @e-mail: diegogoncalves.silva@inf.ufrgs.br
  
  this framework works only with mwe compound of two constituints
  this framework works only with binaries dataset I -> idiomatic L -> Literal
  all data are at the same folder
  Use sentence_location from self.mwe_windows to split data in dev and test

  Code Convetions:

  __constantName__:  local constant
  CONSTANT        :  global constant

"""

class MWESystem:


  def __init__(self,mwe_file_name,path,type_data_set='xml'):
    self.dataset = None
    self.mwe_file_name = mwe_file_name
    #[folder1][folder2]...[foldern] = [(file_name,position_sentence,target)]
    self.mwe_annotations= {} #a vector of tokens divided into folders
    self.type_dataset=type_data_set
    self.PATH = path
    self.files_path = []
    self.tokens_mwe = set()
    self.mwe_windows = {} #[expression] = [(__target__,__sentence_location__,window_sentence,file_path])...]
    self.mapTokenXtoCol = None


  def setup(self):
    self.parseMWEs()
    self.extractFileNames()
    self.parseData()

    all_tokens = list(self.tokens_mwe)
    all_tokens.sort()

    self.mapTokenXtoCol = dict([(all_tokens[i],i) for i in xrange(len(self.tokens_mwe))])

  def isValidSentence(self,sentence,token=' '):
    if len(sentence) < 30 or token not in sentence:
      return False 

    else:
      return True

  def removeLNlist(self,list_data):
    if len(list_data[-1]) < 2:
      return list_data[0:-1]
    return list_data

  #TESTS COPLETED
  def parseMWEs(self):
      """
        # Since the dataset may be huge. We should not read the whole dataset at once
        # This map is useful because we can read one by one without load everything on the memory
        
        FORMAT:  target mwe folder1/folder2/folder3 .... folderN/filename sentence_location
                 I touch_nerve C/CA/CAD 2989
                 
                 target:  I - Idiomatic
                          L - Literal
                          Q - Indefinide

                 fileName: must be XML or CSV
                 sentence_localtion: Key to find the sentence
                                     XML : definded by a tag
                                     CSV : the index + 1 position

                 mwe: Multi-Word-Expression constituints separated by '_'

        AFTER PARSER
        [folder1][folder2]...[foldern] = [(file_name,position_sentence,target)]


      """

      data_set_file = open(self.mwe_file_name).read().split('\n')

      if self.type_dataset=='csv':
        data_set_file = data_set_file[1:]

      data_set_file = self.removeLNlist(data_set_file)

      for sample in data_set_file:
        __samplesplited__ = sample.split()
        __target__        = __samplesplited__[0]
        __mwexpression__  = __samplesplited__[1]
        __folders__       = __samplesplited__[2].split('/')
        __file_name__     = '%s.%s' % (__folders__[-1],self.type_dataset)
        sentence_location = -1

        try:
          sentence_location = int(__samplesplited__[-1])
        except ValueError:
          print 'Exception'
          print 'Sentence position must be integer not a string: Line::%d' % getframeinfo(currentframe()).lineno
          continue

        
        # if __mwexpression__ not in self.mwe_annotations:
        #     self.mwe_annotations[__mwexpression__] = {}

        folder = __folders__[0]
        if folder not in self.mwe_annotations:#[__mwexpression__]:
          self.mwe_annotations[folder] = {}

        actual_dic = self.mwe_annotations[folder]
        index_folder = 1
        
        while index_folder < len(__folders__):#-1: #since the last folder is the file_name
          folder = __folders__[index_folder]

          if folder not in actual_dic:
              actual_dic[folder] = {}

          actual_dic = actual_dic[folder]

          index_folder += 1
        
        if sentence_location not in actual_dic:
          actual_dic[sentence_location] = []

        actual_dic[sentence_location] = (__mwexpression__,__target__)

  #TESTS COPLETED
  def extractFileNames(self):
    actualPATH = os.getcwd()
    os.chdir(self.PATH)

    for path, dirs, files in os.walk("."):
      if './.' not in path and len(path) > 1:
        file_path = os.listdir(os.getcwd()+path[1:])

        for file_mwe in file_path:
          if self.type_dataset in file_mwe: #it is useful because the os.walk get also the directories names
              self.files_path.append('%s/%s' % (path[2:],file_mwe))
    os.chdir(actualPATH)

      
  #TESTS COPLETED
  def parseTokensSentence(self,sentence,root=True):
    """
       CSV: fala|falar|VFIN.PR.3S.IND
            original|root|POS|

       XML: <w c5="VVD" hw="see" pos="VERB">saw </w>
              postype       root   pos       original

       return only tokens from the sentence

    """ 
    tokens = []

    if self.type_dataset == 'csv':
      constituints = sentence.split()

      for c in constituints:
        if '|' not in c:
          raise Exception('| was not found in sentence line::%d' % getframeinfo(currentframe()).lineno)

        if len(c.split()) < 2:
          raise Exception('sentence is not well formed::%d' % getframeinfo(currentframe()).lineno)

        if root:
          tokens.append(c.split('|')[0])
        else:
          tokens.append(c.split('|')[1])

    else:
      if '<w ' not in sentence or 'hw="' not in sentence:
        return []

      constituints = sentence.split('<w ')
      for c in constituints[1:]:
        if root:
          #print "[[[%s]]]" % c
          tokens.append(c.split('hw="')[1].split('"')[0])
        else:
          tokens.append(c.split('>')[1].split(' </w>')[0])


    return tokens

  def parseXMLfile(self,xml_file_name):
    xml_data = self.removeLNlist(open('%s/%s' %(self.PATH,xml_file_name)).read().split('\n'))
    
    xml_sentences = {}

    for sentence in xml_data:
        if not self.isValidSentence(sentence,'<s n="'):
          continue
        sentence_location = sentence.split('<s n="')[1].split('"')[0]

        if not sentence_location.isdigit():
            print 'sentence location %s from file_name: %s is not numeric Line::%d' % (sentence_location,xml_file_name,getframeinfo(currentframe()).lineno)
            continue
        xml_sentences[int(sentence_location)] = sentence

    
    return xml_sentences

  def getOneWindow(self,expression,dataToken, sentence_location,length=10):
      """
        dataToken  = [sentence_number] = sentence
      """

      wordLeft,wordRight = expression.split('_')[0],expression.split('_')[1]

      if sentence_location not in dataToken:
           raise Exception('A sentenca de numero %d no arquivo Line::%d' % (sentence_location,getframeinfo(currentframe()).lineno))

      sentence_parsed = self.parseTokensSentence(dataToken[sentence_location])


      if wordLeft not in sentence_parsed or wordRight not in sentence_parsed:
        raise Exception('the MWE %s does is not within the data Line::%d' % (expression,getframeinfo(currentframe()).lineno))
      
      left_windows_sentence = sentence_parsed[0:sentence_parsed.index(wordLeft)]
      left_windows_sentence = left_windows_sentence[max(0,len(left_windows_sentence)-10):len(left_windows_sentence)]
      right_windows_sentence = sentence_parsed[sentence_parsed.index(wordRight)+1:]
      right_windows_sentence = right_windows_sentence[0:min(len(right_windows_sentence),10)]
      index = 0

      if len(left_windows_sentence) < length and sentence_location-1 in dataToken:
        previous_sentence = self.parseTokensSentence(dataToken[sentence_location-1])
        previous_sentence = previous_sentence[::-1]
        index = 0
        while len(left_windows_sentence) < length and index < len(previous_sentence):
          left_windows_sentence.insert(0,previous_sentence[index])
          index += 1

      if len(right_windows_sentence) < length and sentence_location+1 in dataToken:

        next_sentence = self.parseTokensSentence(dataToken[sentence_location+1])
        index = 0
        while len(right_windows_sentence) < length and index < len(next_sentence):
          right_windows_sentence.append(next_sentence[index])
          index += 1

      return (left_windows_sentence,right_windows_sentence)

  def parseData(self):
     """
       From every file verify wheter it is within mwe_anottations or not
       TODO: Optimize this
     """
     for file_path in self.files_path:
        #print 'processing...',file_path
        __folders__   = file_path.split('/')[0:-1]
        __datamwe__   = None
        __file_name__ = file_path.split('/')[-1].split('.')[0]

        index_folder = 1
        annotations_mwe = self.mwe_annotations[__folders__[0]]

        
        #navigate into folders
        while index_folder < len(__folders__):
          annotations_mwe = annotations_mwe[__folders__[index_folder]]
          index_folder += 1
        
        if __file_name__ not in annotations_mwe:
          continue

        if self.type_dataset == 'xml':
          __datamwe__ = self.parseXMLfile(file_path)
        else:
          __datamwe__ = self.parseCSVfile()

        for __sentence_location__, data in annotations_mwe[__file_name__].iteritems():
            __mwe_expression__ = data[0]
            __target__         = data[1]
        
            if __target__ not in 'LI':
              continue

            left_windows_sentence,right_windows_sentence = self.getOneWindow(__mwe_expression__,__datamwe__,__sentence_location__)

            self.tokens_mwe = self.tokens_mwe.union(set(right_windows_sentence))
            self.tokens_mwe = self.tokens_mwe.union(set(left_windows_sentence))
            self.tokens_mwe = self.tokens_mwe.union(set(__mwe_expression__.split('_')))

            if __mwe_expression__ not in self.mwe_windows:
                self.mwe_windows[__mwe_expression__] = []

            window_sentence = left_windows_sentence
            window_sentence.extend(right_windows_sentence)

            self.mwe_windows[__mwe_expression__].append((__target__,__sentence_location__,window_sentence,file_path))

  def splitData(self,dev=75,test=25):
      test_data = []
      dev_data  = []

      for exp, datamwe in self.mwe_windows.iteritems():

          for data in datamwe:
              data = list(data)
              data.append(exp)

              sort = random.randint(1,100)
              if sort <= dev:
                  dev_data.append(data)
              else:
                  test_data.append(data)

      return (dev_data,test_data)


  def getTokensFrequency(self,data,__targets_labels__):
      tokens_frequency = {'I':{},'L':{}}
      
      for d in data:
          target            = d[0]
          window_sentence   = d[2]

          if target not in __targets_labels__:
              continue

          for ws in window_sentence:
              if ws not in tokens_frequency[target]:
                  tokens_frequency[target][ws] = 0
              tokens_frequency[target][ws] += 1
      return tokens_frequency 

  def buildMatrix(self,data,__targets_labels__='IL'):
      """
          Output
              X: matriz with the token score for each sentence
              Y: target of each sentence
          
          Input
            [(__target__,__sentence_location__,window_sentence,file_path,expression])...]

      """
      X = []
      Y = []

      tokens_frequency = self.getTokensFrequency(data,__targets_labels__)

      for d in data:
        target            = d[0]
        window_sentence   = d[2]
        X.append([0 for i in xrange(len(self.tokens_mwe))])
        if target == 'I':
            Y.append(1)
        else:
            Y.append(0)

        for ws in window_sentence:
            if ws in tokens_frequency[target]:
                X[-1][self.mapTokenXtoCol[ws]] = tokens_frequency[target][ws]
            else:
                X[-1][self.mapTokenXtoCol[ws]] += 1

      return (X,Y)
  

  def RD_PCA(self,data,components=10):
	X = np.array(data)
	pca = PCA(n_components=components)
	pca.fit(X)

	return list(X)

  def RD_SVD(self,data,components=10):
    svd = TruncatedSVD(components)#, n_iter=7, random_state=42)
    svd.fit(data)  

    return data

  def testSVM(self,dev,test):
      clf = svm.SVC()
      clf.fit(dev['X'], dev['Y'])  
      resultSVM = {1:0,0:0}
      TP = 0.0
      TN = 0.0
      FP = 0.0
      FN = 0.0

      for i in xrange(len(test['X'])):
          result = clf.predict([test['X'][i]])
          predicted = result[0]
          target = test['Y'][i]
          resultSVM[predicted] += 1

          if target == 1:
              if predicted == target:
                  TP += 1
              else:
                  FN += 1
          elif target == 0:
              if predicted == target:
                  TN += 1
              else:
                  FP += 1
      precision = 0
      recall = 0
      F1 = 0

      print 'TP=%.f FP=%.f TN=%.f FN=%f' % (TP,FP,TN,FN)
      
      if TP+FP != 0:
        precision = TP/(TP+FP)
      accuracy  = (TP+TN)/(TP+TN+FP+FN)
      
      if TP+FN != 0:
        recall    = TP/(TP+FN)
      if recall+precision != 0:
        F1 = (2*(recall*precision))/(recall+precision)

      print 'result SVM:',resultSVM
      print precision,accuracy,recall,F1
          

      



if "__main__":
  c = MWESystem('cook_mwe.txt',os.getcwd()+'/dados')
  c.setup()
  train_data,test_data = c.splitData()
  all_data = train_data
  all_data.extend(test_data)
  all_data_x,all_data_y = c.buildMatrix(all_data)

  for components in xrange(5, 100, 5):
  	  print 'trainning %d compoments with SVD' % components
	  all_data = c.RD_SVD(all_data_x,components)
	  x_train,x_test = all_data[0:int(len(all_data)*.75)],all_data[int(len(all_data)*.75):]
	  y_train,y_test  = all_data_y[0:int(len(all_data_y)*.75)],all_data_y[int(len(all_data_y)*.75):]
	 
	  # x_train,y_train = c.buildMatrix(train_data)
	  # x_test,y_test = c.buildMatrix(test_data)
	  print len(x_train),len(y_train)
	  print len(x_test),len(y_test)
	  c.testSVM({'X':x_train,'Y':y_train},{'X':x_test,'Y':y_test})






  