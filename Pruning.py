#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import _tree
import numpy as np

# 加载 iris 数据集
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)

# 构建决策树分类器
clf = DecisionTreeClassifier(random_state=0)

# 训练决策树分类器
clf.fit(X_train, y_train)

# 对测试集进行预测
y_pred = clf.predict(X_test)

# 计算分类器的准确率
accuracy = clf.score(X_test, y_test)
print("Accuracy:", accuracy)


# In[2]:


list_index = []

def index_record(tree,index,X,y,list_index):
    
    list_index.append(index)
    
    if tree.children_left[index] == _tree.TREE_LEAF and  tree.children_right[index] == _tree.TREE_LEAF:
        return
    
    index_record(tree,tree.children_left[index],X,y,list_index)
    
    index_record(tree,tree.children_right[index],X,y,list_index)
    
index_record(clf.tree_,0,X_test,y_test,list_index)

print(list_index)


# In[3]:


code_list = [np.random.randint(0,2) for i in range(len(list_index))]

print(code_list)


# In[4]:


from sklearn.tree import plot_tree

plot_tree(clf)


# In[5]:


def GA_pruning(tree,index,index_list):
    if index_list[tree.children_left[index]] == 1:      
        tree.children_left[index] = -1
        leaf_left = True
        
    elif index_list[tree.children_right[index]] == 1:       
        tree.children_right[index] = -1
        leaf_right = True


    if tree.children_left[index] == -1 and tree.children_right[index] == -1:
        return

    GA_pruning(tree,tree.children_left[index],index_list)
    GA_pruning(tree,tree.children_right[index],index_list)
    
    return


# In[6]:


leaf_or_not = False

GA_pruning(clf.tree_,0,code_list)

plot_tree(clf)


# In[ ]:





# In[ ]:





# In[ ]:




