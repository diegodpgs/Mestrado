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

  __constantName__  :  local constant
  CONSTANT          :  global constant
  _R_thisisaregister:  register

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
    self.conta = 0
    self.stopwords = open('nltk_stopwords.txt').read().split('\n')
    self.log = open('log.log','w')
    self._R_sentencenotfound = 0

  def __findSentenceMWE(self,expression,sentence):
      for i in xrange(len(self.mwe_windows[expression])):
         if self.mwe_windows[expression][i] == sentence:
             return i
      return -1

  def setup(self):
    self.parseMWEs()
    self.extractFileNames()
    self.parseData()
    all_tokens = list(self.tokens_mwe)
    all_tokens.sort()
    self.mapTokenXtoCol = dict([(all_tokens[i],i) for i in xrange(len(self.tokens_mwe))])
    self.log.write('%d sentences were not found\n' % self._R_sentencenotfound)

  def isValidSentence(self,sentence,token=' '):
    if len(sentence) < 8 or token not in sentence:
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
          tokens.append(c.split('>')[1].split('</w>')[0])
          if tokens[-1][-1] == ' ':
            tokens[-1] = tokens[-1][0:-1]


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
  def removeStopWords(self,vec):
      index = 0
      while index < len(vec):
            if vec[index] in self.stopwords:
                 del vec[index]
            else:
                index += 1
      return vec

  def getOneWindow(self,expression,dataToken, sentence_location,length=10,removeStopWords=True):
      """
        dataToken  = [sentence_number] = sentence
      """
      #__analidadas = [sentence_location]
      wordLeft,wordRight = expression.split('_')[0],expression.split('_')[1]

      if sentence_location not in dataToken:
           print 'A sentenca de numero %d no arquivo Line::%d' % (sentence_location,getframeinfo(currentframe()).lineno)
           self.log.write('--------%s:sentence %d not located\n' % ('E'*10,sentence_location))
           self._R_sentencenotfound += 1
           return -1

      sentence_parsed = self.parseTokensSentence(dataToken[sentence_location])

      if wordLeft not in sentence_parsed or wordRight not in sentence_parsed:
        print 'the MWE %s does is not within the data Line::%d' % (expression,getframeinfo(currentframe()).lineno)
        self.log.write('--------%s Sentence %s not not located\n' % ('E'*10,expression))
        return -1
      
      left_windows_sentence  = sentence_parsed[0:sentence_parsed.index(wordLeft)]
      left_windows_sentence  = left_windows_sentence[max(0,len(left_windows_sentence)-length):len(left_windows_sentence)]      
      right_windows_sentence = sentence_parsed[sentence_parsed.index(wordRight)+1:]
      right_windows_sentence = right_windows_sentence[0:min(len(right_windows_sentence),length)]
      ST = []
      ST.append(sentence_location)

      if removeStopWords:
        left_windows_sentence  = self.removeStopWords(left_windows_sentence)
        right_windows_sentence = self.removeStopWords(right_windows_sentence)
      
      if len(left_windows_sentence) < length and sentence_location-1 in dataToken:
        actual_sentence = sentence_location-1

        while len(left_windows_sentence) < length and actual_sentence > 0:
          previous_sentence = self.parseTokensSentence(dataToken[actual_sentence])
          previous_sentence = previous_sentence[::-1]
          index = 0
          while len(left_windows_sentence) < length and index < len(previous_sentence):
            left_windows_sentence.insert(0,previous_sentence[index])
            if removeStopWords and left_windows_sentence[-1] in self.stopwords:
              left_windows_sentence.pop()

            index += 1
          ST.append(actual_sentence)
          actual_sentence -= 1
      self.log.write('%s;%02d;' % (expression,len(left_windows_sentence)))
      if len(right_windows_sentence) < length and sentence_location+1 in dataToken:
        actual_sentence = sentence_location+1
        while len(right_windows_sentence) < length and actual_sentence < len(dataToken):
          next_sentence = self.parseTokensSentence(dataToken[actual_sentence])
          index = 0
          while len(right_windows_sentence) < length and index < len(next_sentence):
            right_windows_sentence.append(next_sentence[index])
            if removeStopWords and right_windows_sentence[-1] in self.stopwords:
              right_windows_sentence.pop()
            index += 1
          ST.append(actual_sentence)
          actual_sentence += 1
      ST.sort()
      self.log.write('%02d;%d;%d\n' % (len(right_windows_sentence),len(ST),ST[-1]-ST[0]))
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
        if __folders__[0] not in self.mwe_annotations:
           continue

        annotations_mwe = self.mwe_annotations[__folders__[0]]
        #print file_path
        
        #navigate into folders
        try:
           while index_folder < len(__folders__):
               annotations_mwe = annotations_mwe[__folders__[index_folder]]
               index_folder += 1
        except:
              print 'the folder %s was not annotated' % __folders__[index_folder]
              pass 
           
        if __file_name__ not in annotations_mwe:
         continue

        if self.type_dataset == 'xml':
         __datamwe__ = self.parseXMLfile(file_path)
        else:
         __datamwe__ = self.parseCSVfile()
        #self.log.write('----%s\n' % file_path)
        for __sentence_location__, data in annotations_mwe[__file_name__].iteritems():
           __mwe_expression__ = data[0]
           __target__         = data[1]
       
           if __target__ not in 'LI':
             continue
           
           
           output = None
           try:
              self.log.write('%s;%d;' % (file_path,__sentence_location__))
              output = self.getOneWindow(__mwe_expression__,__datamwe__,__sentence_location__)
              if type(output) == int:
                 continue
           except:
                print '*'*50
                continue
           left_windows_sentence,right_windows_sentence = output
           #    print 'Some problem with %s and %d' % (__mwe_expression__,__sentence_location__)
           #    continue
           self.tokens_mwe = self.tokens_mwe.union(set(right_windows_sentence))
           self.tokens_mwe = self.tokens_mwe.union(set(left_windows_sentence))
           self.tokens_mwe = self.tokens_mwe.union(set(__mwe_expression__.split('_')))
           self.stopwords.extend(right_windows_sentence)
           self.stopwords.extend(left_windows_sentence)
           if __mwe_expression__ not in self.mwe_windows:
               self.mwe_windows[__mwe_expression__] = []

           window_sentence = left_windows_sentence
           window_sentence.extend(right_windows_sentence)

           self.mwe_windows[__mwe_expression__].append((__target__,__sentence_location__,window_sentence,file_path))
           self.conta += 1

  def splitData(self,dev=70,test=30,balance=.9,minimum_samples=20):
      """
        balance means that any target variable will not be higher than 90%
      """

      test_data = {}
      dev_data  = {}
      for exp, datamwe in self.mwe_windows.iteritems(): ##[expression] = [(__target__,__sentence_location__,window_sentence,file_path])...]
          test_data[exp] = []
          dev_data[exp] = []
          balanced = {'I':0.0,'L':0.0}
          if len(datamwe) < minimum_samples:
             continue
          for data in datamwe:
              data = list(data)
              data.append(exp)
              balanced[data[0]] += 1

              sort = random.randint(1,100)
              if sort <= dev:
                  dev_data[exp].append(data)
              else:
                  test_data[exp].append(data)
          difference = abs(balanced['L']-balanced['I'])
          if abs(difference)/(sum(balanced.values())) > balance:
             test_data[exp] = []
             dev_data[exp] = []
             continue
          
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
            X[-1][self.mapTokenXtoCol[ws]] += 1

      return (X,Y)
  

  def RD_PCA(self,data,components=30):
    X = np.array(data)
    pca = PCA(n_components=components)
    pca.fit(X)
    X = pca.transform(X)

    return list(X)

  def RD_SVD(self,data,components=10):
    svd = TruncatedSVD(components)#, n_iter=7, random_state=42)
    return svd.fit_transform(data)  

  def mutual(self,A,B):
      TP = 0.0
      o = 0
      for i in xrange(len(A)):
        if A[i] != 0:
           o += 1
 
      if A[i] == B[i]:
               TP += 1
      return TP/o

  def testSVM(self,dev,test):
      clf = svm.SVC(kernel='linear')
      clf.fit(dev['X'], dev['Y'])  
      print dev['Y']
      print test['Y']
      resultSVM = {1:0,0:0}
      TP = 0.0
      TN = 0.0
      FP = 0.0
      FN = 0.0
      svmpredicted = []
      for i in xrange(len(test['X'])):
          test['X'][i] = list(test['X'][i])
          dev['X'][i] = list(dev['X'][i])
          result = clf.predict([test['X'][i]])
          #print 'count',dev['X'][i].count(1), test['X'][i].count(1),self.mutual(test['X'][i],dev['X'][i])
          predicted = result[0]
          target = test['Y'][i]
          resultSVM[predicted] += 1
          svmpredicted.append(predicted)
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
      print 'SVM RESULTS %s' % (str(resultSVM))      
      print 'TARGET 0:%d 1:%d' % (test['Y'].count(0),test['Y'].count(1))
      if TP+FP != 0:
        precision = TP/(TP+FP)
      accuracy  = (TP+TN)/(TP+TN+FP+FN)
      
      if TP+FN != 0:
        recall    = TP/(TP+FN)
      if recall+precision != 0:
        F1 = (2*(recall*precision))/(recall+precision)

      #print 'result SVM:',resultSVM
      #for i in xrange(len(svmpredicted)):
      #    print svmpredicted[i],test['Y'][i],svmpredicted[i]==test['Y'][i]
      #print test['Y'].count(1),test['Y'].count(0)

      return precision,accuracy,recall,F1
          

if "__main__":
  c = MWESystem('cook_mwe.txt',os.getcwd()+'/Texts')
  c.setup()
  __runs__ = 10
  resultsW = open('results.txt','w')
  resultsData = {}
  for kx in xrange(__runs__):   
      train_data,test_data = c.splitData()
      tP,tA,tR,tF1 = 0,0,0,0
      P,A,R,F1 = -1,-1,-1,-1
      for exp, d in train_data.iteritems():
          if exp not in resultsData:
             resultsData[exp] = {'P':[],'A':[],'R':[],'F1':[]}
          if len(d) == 0:
             continue
          all_data = d
          all_data.extend(test_data[exp])
          all_data_x,all_data_y = c.buildMatrix(all_data)
          all_data_x = list(c.RD_PCA(all_data_x))   
          
          x_train,x_test = all_data_x[0:int(len(all_data_x)*.75)],all_data_x[int(len(all_data_x)*.75):]
          y_train,y_test  = all_data_y[0:int(len(all_data_y)*.75)],all_data_y[int(len(all_data_y)*.75):]
          
          try:
            RSVM = c.testSVM({'X':list(x_train),'Y':y_train},{'X':list(x_test),'Y':y_test})
            P,A,R,F1 = RSVM
            resultsData[exp]['P'].append(P)
            resultsData[exp]['A'].append(A)
            resultsData[exp]['R'].append(R)
            resultsData[exp]['F1'].append(F1)
            print 'EXP:[%s]       P:%1.2f A:%1.2f R:%1.2f F1:%1.2f' % (exp,P,A,R,F1)
          except:
            resultsData[exp]['P'].append(-1)
            resultsData[exp]['A'].append(-1)
            resultsData[exp]['R'].append(-1)
            resultsData[exp]['F1'].append(-1)
            
  for exp, r in resultsData.iteritems():
      if r['P'] == []:
        continue 
      P = sum(r['P'])-(r['P'].count(-1)*-1)
      P = P/(len(r['P'])-r['P'].count(-1))
      A = sum(r['A'])-(r['A'].count(-1)*-1)
      A = A/(len(r['A'])-r['A'].count(-1))
      R = sum(r['R'])-(r['R'].count(-1)*-1)
      R = R/(len(r['R'])-r['R'].count(-1))
      F1 = sum(r['F1'])-(r['F1'].count(-1)*-1)
      F1 = F1/(len(r['F1'])-r['F1'].count(-1))
      resultsW.write('%s;%1.3f;%1.3f;%1.4f;%1.2f\n' % (exp,P,A,R,F1))
  f = nltk.FreqDist(c.stopwords)
  print f




