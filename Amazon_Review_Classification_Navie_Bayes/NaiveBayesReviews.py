# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 14:03:42 2019

@author: Naveen Kumar
"""        
from collections import OrderedDict   #Ordered Dictionary to store probability of each feature
import pandas as pd                    
import string
# reviews cleaning 
from nltk.corpus import stopwords
STOP_WORDS = set(stopwords.words('english'))
STOP_WORDS.add('')
#method to clean reviews
def clean_review(review):
  exclude = set(string.punctuation)
  review = ''.join(ch for ch in review if ch not in exclude)
  split_sentence = review.lower().split(" ")
  clean = [word for word in split_sentence if word not in STOP_WORDS] #removal of stop words
  return clean

#data frame to store 40 Reviews Training data
df=pd.read_csv("C:\\Users\\LENOVO\\Desktop\\Reviews40.csv")
new_df=df.fillna('not available') #filling of NaN with not available
#features list of 10 keywords or key phrases indicative of buyer’s attitude
features=['excellent','good','comfortable','recommend','perfect','bad','problem','terrible','issues','noise']

#cleaning of reviews with clean_review method

reviews_list=[rev for rev in new_df['Review']]

clean_40review_list=[]
for rl in reviews_list:
    clean_40review_list.append(clean_review(rl))

#frequency of each feature in reviews
for f in features:  
    feature_count=[]
    
    for review in clean_40review_list:
       n=review.count(f)
       if n>=1:
           n=1
       feature_count.append(n)
    new_df[f]= feature_count #data frame with frequency of each feature in reviews


#  listing the positive review indexes to find the corresponding feature count in each review
positive_review=new_df['feedback'].unique()[0]
p=new_df['feedback']==positive_review
positive_df=new_df[p]
positive_index=list(positive_df.index) #positive review indexes
pos_review_count=len(positive_index) #positive review count in 40 reviews

#  listing the negative review indexes to find the corresponding feature count in each review
negative_review=new_df['feedback'].unique()[1]
n=new_df['feedback']==negative_review
negative_df=new_df[n]
negative_index=list(negative_df.index)  #negative review indexes
neg_review_count=len(negative_index) #negative review count in 40 reviews
print(new_df)

#Each feature count in positive reviews
feature_count_pos=[]
for f1 in features:
    f_occurance_p=[]
    for pi in positive_index:
        f_occurance_p.append(new_df.at[pi,f1]) #[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        
        each_feat_count_p=f_occurance_p.count(1)
        
    feature_count_pos.append(each_feat_count_p)
       
print(feature_count_pos)   #count of each positive feature[12, 12, 9, 7, 4, 1, 4, 1, 6, 7]


feature_count_neg=[]
for f1 in features:
    f_occurance_n=[]
    for ni in negative_index:
        f_occurance_n.append(new_df.at[ni,f1])
        
        each_neg_feat_count=f_occurance_n.count(1)
        
    feature_count_neg.append(each_neg_feat_count)        #count of each negative feature [1, 8, 2, 2, 1, 14, 8, 0, 8, 1]
       
print(feature_count_neg) #[0, 0, 0, 0, 0, 0, 0, 10, 0, 0]

############probability of each feature with feedback positive and negative :  𝑝 (𝑥1/Ci),,,,,,,,𝑝 (𝑥1/Ci)

prob_pos=[]
for x in feature_count_pos:
    prob_pos.append(x/pos_review_count)
print(prob_pos) #positive probabilities of each feature[0.6, 0.6, 0.45, 0.35, 0.2, 0.05, 0.2, 0.05, 0.3, 0.35]


prob_neg=[]
for y in feature_count_neg:
    prob_neg.append(y/neg_review_count)
print(prob_neg) #negative probabilities of each feature[0.05, 0.4, 0.1, 0.1, 0.05, 0.7, 0.4, 0.0, 0.4, 0.05]


################ Dictionary with (key:value) as (feature,probability)
class NaveBayesClassifier_Positive(OrderedDict):  #customization the dictionary output
    def __missing__(self,k):
        self[k]=' '
        return self[k]
positive_dic=NaveBayesClassifier_Positive()
i=0
for epf in features:
    positive_dic[epf]=prob_pos[i]
    i=i+1
print(positive_dic)
#NaveBayesClassifier_Positive([('excellent', 0.6), ('good', 0.6), ('comfortable', 0.45), ('recommend', 0.35), 
#('perfect', 0.2), ('bad', 0.05), ('problem', 0.2), ('terrible', 0.05), ('issues', 0.3), ('noise', 0.35)])

class NaveBayesClassifier_Negative(OrderedDict):
    def __missing__(self,k):
        self[k]=' '
        return self[k]
negative_dic=NaveBayesClassifier_Negative()
j=0
print(prob_neg[0])
for enf in features:
    negative_dic[enf]=prob_neg[j]
    j=j+1
print(negative_dic)
#NaveBayesClassifier_Negative([('excellent', 0.05), ('good', 0.4), ('comfortable', 0.1), ('recommend', 0.1), 
#('perfect', 0.05), ('bad', 0.7), ('problem', 0.4), ('terrible', 0.0), ('issues', 0.4), ('noise', 0.05)])

############Validation with TEN reviews################

#data frame with 10 Reviews data
tenreviewsread_df=pd.read_csv("C:\\Users\\LENOVO\\Desktop\\TenReviews.csv")
tenreviews_df=tenreviewsread_df.fillna('not available')
#print(tenreviews_df)
list_rev=[r for r in tenreviews_df['Reviews10']]
clean_10review_list=[]
for lr in list_rev:
    clean_10review_list.append(clean_review(lr))


pb_pos_review=pos_review_count/(neg_review_count+pos_review_count) #probability that a review is positive of 40 reviews
pb_neg_review=neg_review_count/(neg_review_count+pos_review_count) #probability that a review is negative of 40 reviews

##### positive probabilities of ten reviews
pb_pos=[]
for er in clean_10review_list:
    m=1
    for f in features:
        n=er.count(f)
        if n>=1:
            m=m*float(positive_dic[f])
    pb_pos.append(m*pb_pos_review)
    
tenreviews_df['Postive_pb']=pb_pos

#### negative probabilities of ten reviews

pb_neg=[]
for ern in clean_10review_list:
    m1=1
    for f1 in features:
        n1=ern.count(f1)
        if n1>=1:
            m1=m1*float(negative_dic[f1])
    pb_neg.append(m1*pb_neg_review)
    
tenreviews_df['Negative_pb']=pb_neg


### Classification of Feedbacks based on probabilities
u=0
validated_feedback=[]
for er in clean_10review_list:
    
    if (float(pb_pos[u]) > float(pb_neg[u])): 
        validated_feedback.append('positive')
        
    else:
        validated_feedback.append('negative')
    u=u+1
tenreviews_df['ValidatedFeedback']=validated_feedback 
actual_feedback=tenreviews_df['ActualFeedback']

print(tenreviews_df)

##### Accuracy of classifier 
v=0
pr=0
for er in clean_10review_list:
    if validated_feedback[v] == actual_feedback[v]: #comparison of positive and negative feedback in actual and validated feedbacks
        pr=pr+1
    v=v+1
print('Accuracy of the classifier:',pr/10)

