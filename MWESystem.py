import time
import sys
import os
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import numpy as np
from inspect import currentframe, getframeinfo


"""
  @author: Diego Pedro
  @e-mail: diegogoncalves.silva@inf.ufrgs.br

  Code Convetions:

  UPERCASE:  variables which does not change 
             OBS. It is note used locally in small methods.
  lowercase: variables may change

"""

class MWESystem:


  def __init__(self,mwe_file_name,type_data_set='xml'):
    self.dataset = None
    self.mwe_file_name = mwe_file_name
    self.sentences = {} #a vector of tokens divided into folders
    self.type_dataset=type_data_set;


  def removeLNlist(self,list_data):
    if len(list_data[-1]) < 2:
      return list_data[0:-1]
    return list_data
    
  def parseCookMWE(self):
      """
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


      """

      data_set_file = open(self.mwe_file_name).read().split('\n')

      if self.type_dataset=='csv':
        data_set_file = data_set_file[1:]

      data_set_file = self.removeLNlist(data_set_file)

      for sample in data_set_file:
        sample_splited = sample.split()
        target = sample_splited[0]
        mw_expression = sample_splited[1]
        folders = sample_splited[2].split('/')
        file_name = '%s.%s' % (folders[-1],self.type_dataset)
        sentence_position = -1

        try:
          sentence_position = int(sample_splited[-1])
        except ValueError:
          frameinfo = getframeinfo(currentframe())
          print 'Exception'
          print 'Sentence position must be integer not a string: Line::%d' % frameinfo.lineno
          continue

        
        if mw_expression not in self.sentences:
            self.sentences[mw_expression] = {}


        
        folder = folders[0]
        if folder not in self.sentences[mw_expression]:
          self.sentences[mw_expression] = {folder:{}}

        actual_dic = self.sentences[mw_expression][folder]
        index_folder = 1
        
        while index_folder < len(folders)-1: #since the last folder is the file_name
          folder = folders[index_folder]

          if folder not in actual_dic:
              actual_dic[folder] = {}

          if index_folder+2 == len(folders):
            if type(actual_dic[folder]) == dict:
                actual_dic[folder] = []
          else:
            actual_dic = actual_dic[folder]

          index_folder += 1

        actual_dic[folder].append((file_name,sentence_position,target))
      

# #L blow_smoke A/AD/ADA 693
# #[A][AD][ADA][693] = [blow_moke,L]
# def mapToSentence(cookMWE):
#   sentence_map = {}

#   for expression, data in cookMWE.iteritems():
#     for folder, data2 in data.iteritems():
#       #print folder
#       for celula in data2:
#         folder3 = celula[0].split('/')[-1]
#         folder2 = celula[0].split('/')[0]
#         sentence = celula[1]
#         label = celula[2]

#         if folder not in sentence_map:
#           sentence_map[folder] = {}
        
#         if folder2 not in sentence_map[folder]:
#           sentence_map[folder][folder2] = {}

#         if folder3 not in sentence_map[folder][folder2]:
#           sentence_map[folder][folder2][folder3] = {}

#         if sentence not in sentence_map[folder][folder2][folder3]:
#           sentence_map[folder][folder2][folder3][sentence] = []
#         sentence_map[folder][folder2][folder3][sentence] = [expression,label]
#     #print 'expression ',expression,' mapped to sentence'
#   return sentence_map

# def parseRootTokens(sentence): 
#   rooted = sentence.split('hw="')
#   roots = []
#   for s in rooted:
#     roots.append(s.split('"')[0])
#   return roots[1:]

# def parseOriginalTokens(sentence): 
#   rooted = sentence.split('</w>')
#   roots = []
#   for s in rooted:
#     roots.append(s.split('>')[-1])
#   return roots[:-1]


# def getOneWindow(expA,expB,xml_data,number,length):
#     if number not in xml_data:
# 	return -1,-1

#     sentence_root = parseRootTokens(xml_data[number])

#     if expA not in sentence_root or expB not in sentence_root:
#       return None,None

#     left_windows_sentence = sentence_root[0:sentence_root.index(expA)]
#     left_windows_sentence = left_windows_sentence[0:min(len(left_windows_sentence),10)]
#     right_windows_sentence = sentence_root[sentence_root.index(expB)+1:]
#     right_windows_sentence = right_windows_sentence[0:min(len(right_windows_sentence),10)]
#     index = 0

#     if len(left_windows_sentence) < length and number-1 in xml_data:
#       previous_sentence = parseRootTokens(xml_data[number-1])
#       previous_sentence = previous_sentence[::-1]
#       index = 0
#       while len(left_windows_sentence) < length and index < len(previous_sentence):
#         left_windows_sentence.insert(0,previous_sentence[index])
#         index += 1

#     if len(right_windows_sentence) < length and number+1 in xml_data:

#       next_sentence = parseRootTokens(xml_data[number+1])
#       index = 0
#       while len(right_windows_sentence) < length and index < len(next_sentence):
#         right_windows_sentence.append(next_sentence[index])
#         index += 1

#     return (left_windows_sentence,right_windows_sentence)

# def parseSentence(file_xml_name):
#   xml_data = open('%s' % file_xml_name).read().split('\n')[2:]
#   #try:
#   d = {}
#   for line in xml_data[1:-1]:
#       if len(line) < 30 or '<s n="' not in line:
# 	continue
#       try:
#       	sentence_number = int(line.split('<s n="')[1].split('"')[0])
# 	d[sentence_number] = line
#       except:
# 	continue
#   #print '%.3f %% VALID SENTENCES' % (len(d.keys())/float(len(xml_data)))
#   return d


# #[exp]= [(label,words_s1),words_s2...]
# #per-expression
# #TOKENS[Label] = {word1: 3, Word2>5}
# def getWindows(windows,tokens,mapSentence,file_xml_name,length=10):
#   expressions = set()
#   #print file_xml_name
#   folder1, folder2,file_xml = file_xml_name.split('/')[-3],file_xml_name.split('/')[-2],file_xml_name.split('/')[-1].split('.')[0]
  
#   xml_data = parseSentence(file_xml_name)
  
#   if xml_data == None:
#     return None
#   #print 'Running.....'
#   if (folder1 not in mapSentence) or (folder2 not in mapSentence[folder1]) or (file_xml not in mapSentence[folder1][folder2]):
#     return set()
#   sentences_mwe = mapSentence[folder1][folder2][file_xml]
#   #print 'running'
#   for number, data in sentences_mwe.iteritems():
#     expressions.add(data[0])
#     expA,expB = data[0].split('_')[0],data[0].split('_')[1]
#     label = data[1]
#     if label not in 'LI':
#       continue

#     left_windows_sentence,right_windows_sentence = getOneWindow(expA,expB,xml_data,number,length)
#     if left_windows_sentence == -1:
#         print 'problem with sentence %d not find within file %s' % (number,file_xml_name)
#         continue
#     # tokens = tokens.union(set(right_windows_sentence))
#     # tokens = tokens.union(set(left_windows_sentence))
#     # tokens.add(data[0])

#     if data[0] not in windows:
#       windows[data[0]] = []
#     window_sentence = left_windows_sentence
#     window_sentence.extend(right_windows_sentence)

#     windows[data[0]].append((label,window_sentence))
#     for ws in window_sentence:
#       if ws not in tokens[label]:
#         tokens[label][ws] = 0
#       tokens[label][ws] += 1

#   return expressions

# def getFiles(FOLDER_DATA):
#   files = []
#   PATH = os.getcwd()+'/'+FOLDER_DATA
#   for folder in os.listdir(PATH):
#     if '.DS_Store' in folder:
#       continue
#     for subfolder in os.listdir('%s/%s' % (PATH,folder)):
#       if '.DS_Store' in subfolder:
#         continue
#       for file_xml_name in os.listdir('%s/%s/%s' % (PATH,folder,subfolder)):
#         SUB_PATH = '%s/%s/%s' % (PATH,folder,subfolder)
#         files.append('%s/%s' % (SUB_PATH,file_xml_name))
#   return files

# def getData(mapSentence,files):
  
#   windows = {}
#   tokens = {'L':{},'I':{}}
#   expressions = set()
#   count = 0.0
#   for file_name in files:
#     count += 1
#     if count % 50 ==0:
# 	print '%.3f' % (count/len(files))
#     r = getWindows(windows,tokens,mapSentence,file_name)
#     if r == None:
#       continue
#     #print 'SIZE WINDOWS',len(windows.keys())#print 'Windows computed for the file',file_name
    
#   return (windows,tokens)

# #MWE,[0,0,0],LABEL[0-literal,1-idiomatic]
# def normalizeVectors(windows,tokens,exp):
#   #[exp]= [(label,words_s1),words_s2...]
#   win_exp = windows[exp]
#   tokens_total = set(tokens['L'].keys())
#   tokens_total = list(tokens_total.union(set(tokens['I'].keys())))
#   vectorMAPindex = dict([(tokens_total[i],i) for i in xrange(len(tokens_total))])

#   #expression, ocorrence,label
#   MATRIX = []
#   labels = []
  
#   for ds in win_exp:
#       label,words = ds[0],ds[1]

#       if label not in 'LI':
#         continue

#       MATRIX.append([0 for i in xrange(len(tokens_total))])

#       for word in words:
#         indexTotoken = vectorMAPindex[word]
#         MATRIX[-1][indexTotoken] = tokens[label][word]

#       if label == 'I':
#         labels.append(1)
#       else:
#         labels.append(0)

#   return (MATRIX,labels)

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
  c = MWESystem('cook_mwe.txt')
  c.parseCookMWE()
  print c.sentences


