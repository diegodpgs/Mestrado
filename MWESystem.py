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
    self.mwe_windows = {} #[expression] = [(__target__,__sentence_location__,window_sentence,file_path])...]


  def setup(self):
    self.parseMWEs()
    self.extractFileNames()
    self.parseData()

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
        print 'processing...',file_path
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

            if __mwe_expression__ not in self.mwe_windows:
                self.mwe_windows[__mwe_expression__] = []

            window_sentence = left_windows_sentence
            window_sentence.extend(right_windows_sentence)

            self.mwe_windows[__mwe_expression__].append((__target__,__sentence_location__,window_sentence,file_path))

  def splitDataByExpression(self,expression,dev=75,test=25):
      test_data = []
      dev_data  = []

      for data in self.mwe_windows[expression]:
          data = list(data)
          data.append(expression)

          sort = random.randint(1,100)
          if sort <= dev:
              dev_data.append(data)
          else:
              test_data.append(data)

      return (test_data,dev_data)


  def getTokensFrequency(self,data,__targets_labels__):
      tokens_frequency = {'I':{},'L':{}}
      
      for d in data:
          target            = d[0]
          window_sentence   = d[2]

          if target not in __targets_labels__:
          	continue

          for ws in window_sentence:
          	if ws not in tokens[target]:
          		tokens[target][ws] = 0
          	tokens[target][ws] += 1
       return tokens 

  def buildMatrixTrain(self,data_dev,__targets_labels__='IL'):
      """
          Output
              X: matriz with the token score for each sentence
              Y: target of each sentence
              mapTokenXtoCol
          
          Input
            [(__target__,__sentence_location__,window_sentence,file_path,expression])...]

      """
      X = [[0 for i in xrange(len(data_dev))] for j in xrange(len(data_dev))]
      tokens_train_frequency = self.getTokensFrequency(data_dev,__targets_labels__)
      
      


# #proxima etapa, normalizar os dados para os vetores
# def test(mapSentence,data_train,data_test):
#   #[exp][L] = {word1:0, word2:1, word3:3}
#   w_train,tokens = train(mapSentence,data_train)
  
#   TP = 0
#   TN = 0
#   FP = 0
#   FN = 0

#   for file_name in data_test:
#     folder1, folder2,file_xml = file_name.split('/')[6],file_name.split('/')[7],file_name.split('/')[8].split('.')[0]
    
#     xml_data = parseSentence(file_xml_name)
#     if xml_data == None:
#       return None

#     if (folder1 not in mapSentence) or (folder2 not in mapSentence[folder1]) or (file_xml not in mapSentence[folder1][folder2]):
#       return set()

#     sentences_mwe = mapSentence[folder1][folder2][file_xml]
#     tokens = set()

#     contagem = {'L':0,'I':0}
    
#     for number, data in sentences_mwe.iteritems():
#       if data[0] not in w_train:
#         continue

#       expA,expB = data[0].split('_')[0],data[0].split('_')[1]
#       label = data[1]

#       left_windows_sentence,right_windows_sentence = getOneWindow(expA,expB,xml_data,number,length)
#       tokens_test = left_windows_sentence
#       tokens_test.extend(right_windows_sentence)
#       tokens_test = list(set(tokens_test))
#       for t in tokens_test:
#         if t in w_train[data[0]]['L']:
#           contagem['L'] += w_train[data[0]]['L'][t]

#         if t in w_train[data[0]]['I']:
#           contagem['I'] += w_train[data[0]]['I'][t]


# def testSVM(svm_,X_test,Y_test):
#   TP = 0.0
#   TN = 0.0
#   FP = 0.0
#   FN = 0.0
#   svmr = {1:0,0:0,-1:0}
#   for indice in xrange(len(X_test)):
#     result = list(svm_.predict([X_test[indice]]))
#     print result
#     #svmr[result] += 1
#     if result == Y_test[indice]:
#       if result == 0:
#         TN += 1
#       else:
#         TP += 1
#     else:
#       if result == 0:
#         FN += 1
#       else:
#         FP += 1
#   precision = 0
#   recall = 0
#   F1 = 0
#   print TP,FP,TN,FN
#   if TP+FP != 0:
#     precision = TP/(TP+FP)
#   accuracy  = (TP+TN)/(TP+TN+FP+FN)
#   print svmr
#   if TP+FN != 0:
#     recall    = TP/(TP+FN)
#   if recall+precision != 0:
#     F1 = (2*(recall*precision))/(recall+precision)
#   return precision,accuracy,recall,F1


# cookparsed = parseCookMWE()
# mapSentence = mapToSentence(cookparsed)
# files = getFiles('Texts')
# W,Tokens = getData(mapSentence,files)
# logr = open('logresults.log','w')
# logr.write('P;A;R;F1;SAMPLES;EXP;ERROR\n')
# sucesso = 0
# fail = 0
# print len(W.keys())
# for exp, data in W.iteritems():
#   X,Y =  normalizeVectors(W,Tokens,exp)
#   print 'normalizeVectors computed for the expression',exp
#   if X == None:
#     print X 
#     continue
#   X = StandardScaler().fit_transform(X)
#   pca = PCA(n_components=200)
#   principalComponents = pca.fit_transform(X)
#   X_train = np.array(principalComponents)
#   X_train,X_test = X_train[0:int(len(X_train)*.75)],X_train[int(len(X_train)*.75):]
#   Y_train,Y_test = Y[0:int(len(Y)*.75)],Y[int(len(Y)*.75):]
#   Y_test = np.array(Y_test)
#   clf = svm.SVC()
#   try:
#     clf.fit(X_train,Y_train)
#     P,A,R,F1 = testSVM(clf,X_test,Y_test)
#     logr.write('%.4f;%.4f;%.4f;%.4f;%d;%s;%s\n' % (P,A,R,F1,len(Y_test),exp,'NO ERROR'))
#     sucesso += 1

#   except ValueError:
#     logr.write('-1;-1;-1;-1;%d;%s;%s\n' % (len(Y_test),exp,'The number of classes has to be greater than one'))
#     print 'A expressao %s nao foi executada pelo SVM devido ao tamanho = %d' % (exp,len(Y_test))
#     fail += 1
#   except KeyError:
#     logr.write('-1;-1;-1;-1;%d;%s;KeyError\n' % (len(Y_test),exp))
#     print 'algum problema de chave ocorreu'
#     fail += 1
#   except TypeError:
#     logr.write('-1;-1;-1;-1;%d;%s;TypeError\n' % (len(Y_test),exp))
#     print 'algum problema de tipo ocorreu'
#     fail += 1
#   except:
#     logr.write('-1;-1;-1;-1;%d;%s;NO IDEA\n' % (len(Y_test),exp))
#     print 'Outro erro nao explicado'
#     fail += 1

# print '%d/%d %.3f' % (sucesso,fail,float(sucesso)/(fail+sucesso))


if "__main__":
  c = MWESystem('cook_mwe.txt',os.getcwd()+'/Texts')
  c.setup()
  train,test = c.splitDataByExpression('blow_smoke')



